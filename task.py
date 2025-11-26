"""fed-avg: Versão otimizada de uma aplicação Flower/PyTorch com dataset NIH Chest X-ray."""

from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter
import gc


# ======================
# Transformações
# ======================
def get_transform(train=True):
    """Retorna transformações com data augmentation para treino."""
    if train:
        return Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.2, contrast=0.2),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
    else:
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])


# ======================
# Custom Dataset for NIH Chest X-ray
# ======================
dataset = None
_patient_partitions = {}
_partition_datasets = {}
_global_df = None

class NIHChestXrayDataset(Dataset):
    """Custom dataset para imagens de raio-x do NIH."""

    def __init__(self, csv_file, image_dirs, transform=None, train_mode=True, indices=None):
        # ✅ Usar o cache global em vez de recarregar
        global _global_df
        
        if _global_df is not None and indices is not None:
            # Usar cache global filtrado
            self.df = _global_df.loc[list(indices)].reset_index(drop=True)
        elif indices is not None:
            # Fallback: carregar apenas linhas necessárias
            self.df = pd.read_csv(csv_file, usecols=[
                'Image Index',
                'Finding Labels',
                'Patient ID',
            ], skiprows=lambda x: x not in indices and x != 0)
        else:
            # Carregar tudo (não recomendado)
            self.df = pd.read_csv(csv_file, usecols=[
                'Image Index',
                'Finding Labels',
                'Patient ID',
            ])
        
        self.df.dropna(subset=['Image Index', 'Finding Labels', 'Patient ID'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.image_dirs = [Path(d) for d in image_dirs]
        self.train_mode = train_mode
        self.transform = transform if transform else get_transform(train_mode)

        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        # ✅ Cache de imagens: criar APENAS se não existir
        if not hasattr(NIHChestXrayDataset, '_image_paths_cache'):
            print("🗂️  Criando cache de caminhos de imagens (uma única vez)...")
            NIHChestXrayDataset._image_paths_cache = {}
            for img_dir in self.image_dirs:
                folder = img_dir / "images"
                if folder.exists():
                    for fname in os.listdir(folder):
                        NIHChestXrayDataset._image_paths_cache[fname] = folder / fname
            print(f"✅ Cache criado: {len(NIHChestXrayDataset._image_paths_cache)} imagens indexadas")
        
        self.image_paths = NIHChestXrayDataset._image_paths_cache

        # ✅ Pré-computar labels
        print(f"🏷️  Pré-processando {len(self.df)} labels...")
        self.label_tensors = self._precompute_labels()

    def _precompute_labels(self):
        """Vetoriza criação de labels para melhor performance."""
        labels_list = []
        disease_to_idx = {d: i for i, d in enumerate(self.diseases)}
        
        for labels_str in self.df['Finding Labels']:
            label = torch.zeros(len(self.diseases))
            if labels_str != 'No Finding':
                for finding in labels_str.split('|'):
                    idx = disease_to_idx.get(finding)
                    if idx is not None:
                        label[idx] = 1
            labels_list.append(label)
        
        return labels_list

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = self.image_paths.get(img_name)

        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.label_tensors[idx]


# ======================
# Modelo CNN
# ======================
class Net(nn.Module):
    """CNN simples para imagens de raio-x (224x224) com saída multi-label."""

    def __init__(self, num_classes=14):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ======================
# Particionamento de dados por paciente
# ======================
dataset = None
_patient_partitions = {}
_partition_datasets = {}  # ✅ Cache de datasets por partição

# ========================
# Secção incluida para realizar balanceamento da base de dados	
# ========================

def filter_balanced_indices(df, max_samples_per_class=50):
    """
	Seleciona homogeneamente as classes, colocando um limite máximo 'max_samples_pre_class'.
	Percorre o DF e pega a imagem se ela tiver uma doença 
    	que ainda não atingiu a cota
    """
    print(f" Limitando dataset a aprox. {max_samples_per_class} amostras por classe...")
    
    # Lista de doenças
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # Contadores
    counts = {d: 0 for d in diseases}
    selected_indices = []
    
    # Embaralhar para garantir seleção aleatória e não pegar sempre os primeiros
    shuffled_indices = np.random.permutation(df.index.values)

    for idx in shuffled_indices:
        labels_str = df.at[idx, 'Finding Labels']
        
        # Se for "No Finding" (sem doença), podemos limitar também ou pular
        if labels_str == 'No Finding':
            continue 

        labels = labels_str.split('|')
        
        # Verifica se essa imagem é útil (se contém alguma doença que ainda precisamos)
        is_useful = False
        for label in labels:
            if label in counts and counts[label] < max_samples_per_class:
                is_useful = True
                break
        
        # Se for útil, adicionamos aos selecionados e atualizamos os contadores
        if is_useful:
            selected_indices.append(idx)
            for label in labels:
                if label in counts:
                    counts[label] += 1
                    
        # Opcional: Parar se todas as cotas estiverem cheias (otimização)
        if all(c >= max_samples_per_class for c in counts.values()):
            break

    print(f" Dataset balanceado: {len(selected_indices)} imagens selecionadas.")
    print(f"   Contagem final (amostra): {counts}")
    
    return selected_indices

# ===================================
# Modificação feita na classe 'load_data'
#	Ao criar o dataset não usamos mais 'client_patients ' mas sim 'balanced_indices', 
#     onde armazenamos a quantidade escolhida de imagens de cada classe
# ===================================


def load_data(partition_id: int, num_partitions: int):
    """Carrega partição do dataset de forma otimizada."""
    global dataset, _patient_partitions, _partition_datasets

    # ✅ OTIMIZAÇÃO 4: Retornar cache se já carregado
    if partition_id in _partition_datasets:
        print(f"Usando cache para partição {partition_id}")
        return _partition_datasets[partition_id]

    base_path = Path(__file__).parent / "archive"
    csv_file = base_path / "Data_Entry_2017.csv"
    image_dirs = [base_path / f"images_{i:03d}" for i in range(1, 13)]

    # ✅ OTIMIZAÇÃO 5: Particionar antes de carregar o dataset completo
    if not _patient_partitions:
        print("Criando particionamento por paciente...")
        # Ler apenas Patient ID para particionar
        patient_df = pd.read_csv(csv_file, usecols=['Patient ID'])
        patient_ids = patient_df['Patient ID'].unique()
        
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        
        splits = np.array_split(patient_ids, num_partitions)
        _patient_partitions = {i: set(splits[i]) for i in range(num_partitions)}
        del patient_df  # Liberar memória

    # ✅ OTIMIZAÇÃO 6: Criar dataset apenas com dados do cliente
    print(f"Carregando dados para partição {partition_id}...")
    full_df = pd.read_csv(csv_file, usecols=['Image Index', 'Finding Labels', 'Patient ID'])
    client_patients = _patient_partitions[partition_id]
    
    # Filtrar apenas pacientes deste cliente
    client_df = full_df[full_df['Patient ID'].isin(client_patients)]

    #Modificação implementada para balanceamento da base, definimos o numero maximo de imagens por classe
    #Então geramos o dataset pros clientes
    # Define quantas imagens por classe
    MAX_IMGS_POR_CLASSE = 50 
    
    # Obtemos os índices originais do DataFrame que correspondem à nossa seleção balanceada
    balanced_indices = filter_balanced_indices(client_df, max_samples_per_class=MAX_IMGS_POR_CLASSE)
    
    del full_df 
    
    # Criar dataset APENAS com os índices selecionados
    client_dataset = NIHChestXrayDataset(
        csv_file=csv_file,
        image_dirs=image_dirs,
        transform=get_transform(train=True),
        train_mode=True,
        indices=set(balanced_indices) # Passamos a lista filtrada
    )

    # Split treino/teste e DataLoaders (Código padrão, sem sampler complexo)
    n = len(client_dataset)
    train_size = int(0.8 * n)
    test_size = n - train_size
    
    train_set, test_set = random_split(
        client_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    use_pin_memory = torch.cuda.is_available()
    num_workers = 2 if torch.cuda.is_available() else 0
    
    trainloader = DataLoader(
        train_set, 
        batch_size=32, 
        shuffle=True, # Volta a ser True pois filtramos os dados antes
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    )
    
    testloader = DataLoader(
        test_set, 
        batch_size=32,
        num_workers=num_workers, 
        pin_memory=use_pin_memory
    )
    
    _partition_datasets[partition_id] = (trainloader, testloader)
    
    gc.collect()
    return trainloader, testloader

# ======================
# Treino / Teste
# ======================
def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((14,), 21.0, device=device))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0

    total_batches = 0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1
            
            # Liberar memória de tensores não utilizados
            del images, labels, outputs, loss
    
    # Forçar coleta de lixo após treinamento
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return running_loss / total_batches if total_batches > 0 else 0.0


def test(net, testloader, device):
    net.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((14,), 21.0, device=device))
    net.eval()

    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            total_correct += torch.sum(preds.eq(labels)).item()
            total += labels.numel()
            
            # Liberar memória
            del images, labels, outputs, probs, preds

    # Coleta de lixo após teste
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    accuracy = total_correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ======================
# Pesos do modelo
# ======================
# ✅ CORREÇÃO 3: Retornar lista de arrays NumPy (formato esperado pelo Flower)
def get_weights(net):
    """Extrai os pesos do modelo como lista de arrays NumPy."""
    return [val.cpu().numpy() for val in net.state_dict().values()]


def set_weights(net, parameters):
    """Define os pesos do modelo a partir de uma lista de arrays NumPy."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

