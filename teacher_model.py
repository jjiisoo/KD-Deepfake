import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
from tqdm import tqdm

# Configuration 설정
CFG = {
    'train_dir': '/home/work/jisoo/Deepfake/train',
    'test_dir': '/home/work/jisoo/Deepfake/test',
    'val_dir': '/home/work/jisoo/Deepfake/valid',
    'batch_size': 32,
    'img_size': 224,
    'resize_size': 256,
    'learning_rate': 1e-4,
    'epochs': 20,
    'dropout': 0.3,
    'fc_hidden_dim': 512,
    'temperature': 4.0,
    'alpha': 0.4,
    'patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'teacher_model_path': 'efficientb7_teacher.pth',
    'save_student_name': 'resnet8_student.pth'
}

# Data Transformation
train_transforms = transforms.Compose([
    transforms.Resize(CFG['resize_size']),
    transforms.CenterCrop(CFG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_val_transforms = transforms.Compose([
    transforms.Resize(CFG['resize_size']),
    transforms.CenterCrop(CFG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
train_dataset = datasets.ImageFolder(root=CFG['train_dir'], transform=train_transforms)
test_dataset = datasets.ImageFolder(root=CFG['test_dir'], transform=test_val_transforms)
val_dataset = datasets.ImageFolder(root=CFG['val_dir'], transform=test_val_transforms)

train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CFG['batch_size'], shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False)

# Device 설정
device = torch.device(CFG['device'])

# Teacher 모델 로드 (EfficientNet-b7)
teacher_model = models.efficientnet_b7(pretrained=True)
num_features = teacher_model.classifier[1].in_features
teacher_model.classifier = nn.Sequential(
    nn.Linear(num_features, CFG['fc_hidden_dim']),
    nn.ReLU(),
    nn.Dropout(CFG['dropout']),
    nn.Linear(CFG['fc_hidden_dim'], 1),
    nn.Sigmoid()
)
teacher_model.load_state_dict(torch.load(CFG['teacher_model_path']))
teacher_model = teacher_model.to(device)
teacher_model.eval()

# Basic Block 정의
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut Connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out

# ResNet8 정의
class ResNet8(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet8, self).__init__()
        self.in_channels = 16

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)

        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # 수정: num_classes 사용
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# 모델 초기화
student_model = ResNet8(num_classes=1).to(device)

'''
def count_model_parameters(model):
    """
    모델의 총 파라미터 수와 학습 가능한 파라미터 수를 출력하는 함수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# ResNet8 모델의 파라미터 수 출력
count_model_parameters(student_model)
'''

def distillation_loss(student_output, teacher_output, ground_truth, temperature, alpha):
    # Soft Targets: MSE Loss 사용
    soft_loss = nn.MSELoss()(
        student_output / temperature,
        teacher_output / temperature
    )
    # Hard Targets: BCEWithLogitsLoss 사용
    hard_loss = nn.BCEWithLogitsLoss()(student_output.squeeze(), ground_truth)
    return alpha * soft_loss + (1 - alpha) * hard_loss


# Precompute Teacher Outputs
print("Precomputing Teacher Outputs...")
precomputed_teacher_outputs = []
with torch.no_grad():
    for inputs, _ in tqdm(train_loader, desc="Precomputing"):
        inputs = inputs.to(device)
        outputs = teacher_model(inputs)
        precomputed_teacher_outputs.append(outputs.cpu())

def train_student_with_precomputed(student, teacher_model, train_loader, val_loader, cfg):
    criterion = nn.BCEWithLogitsLoss()  # 더 안정적인 손실 함수
    optimizer = optim.Adam(student.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg['epochs']):
        print(f"Epoch {epoch + 1}/{cfg['epochs']}")
        student.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for inputs, targets in train_loader_tqdm:
            inputs, targets = inputs.to(cfg['device']), targets.to(cfg['device']).float()

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs).view(-1, 1)  # Teacher 출력 배치별 계산

            student_outputs = student(inputs).view(-1, 1)
            loss = distillation_loss(student_outputs, teacher_outputs, targets, cfg['temperature'], cfg['alpha'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        val_loss = 0.0
        student.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(cfg['device']), targets.to(cfg['device']).float()
                teacher_outputs = teacher_model(inputs).view(-1, 1)
                student_outputs = student(inputs).view(-1, 1)
                val_loss += distillation_loss(student_outputs, teacher_outputs, targets, cfg['temperature'], cfg['alpha']).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student.state_dict(), cfg['save_student_name'])
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print("Early stopping triggered!")
                break




import time
from sklearn.metrics import precision_score, recall_score, f1_score

# 모델 평가 함수
def evaluate_model(model, dataloader, cfg):
    model.eval()
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(cfg['device']), targets.to(cfg['device']).float()
            outputs = model(inputs).squeeze()
            
            # 출력 크기 확인
            if outputs.dim() == 1:  # 차원이 1인 경우
                outputs = outputs.view(-1, 1)
            if targets.dim() == 1:  # 차원이 1인 경우
                targets = targets.view(-1, 1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 이진 분류 결과
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# FPS와 F1-Score 계산 함수
def evaluate_realtime(model, dataloader, device):
    model.eval()
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    start_time = time.time()  # FPS 측정 시작
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Realtime Evaluation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs).squeeze()
            
            # 출력 크기 확인
            if outputs.dim() == 1:  # 차원이 1인 경우
                outputs = outputs.view(-1, 1)
            if targets.dim() == 1:  # 차원이 1인 경우
                targets = targets.view(-1, 1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions = (outputs > 0.5).float()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    end_time = time.time()  # FPS 측정 종료
    start
    '''
    for i in range(100):
        output = model(input)
    end_time
    time = end-start
    fps = 100 / time
    '''
    # FPS 계산
    num_samples = len(dataloader.dataset)
    total_time = end_time - start_time
    fps = num_samples / total_time

    # F1-Score 계산
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    # 평균 Loss 계산
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, fps, precision, recall, f1


# 학습 실행
print("Training Student Model with AKD...")
train_student_with_precomputed(student_model, teacher_model, train_loader, val_loader, CFG)
'''
# 테스트 평가
test_loss, test_accuracy = evaluate_model(student_model, test_loader, CFG)
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

# FPS와 F1-Score 계산
test_loss_realtime, fps, precision, recall, f1 = evaluate_realtime(student_model, test_loader, CFG['device'])
print(f"Realtime Evaluation:")
print(f"Test Loss: {test_loss_realtime:.4f}")
print(f"FPS: {fps:.2f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
'''
