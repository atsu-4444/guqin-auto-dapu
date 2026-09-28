import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import joblib
import torch.nn.functional as F

class EfficientNetSVMClassifier:
    def __init__(self, backbone_path, svm_path, device='cpu', num_classes=203):
        """
        Initialize the hybrid classifier: EfficientNet-B0 (Feature Extraction) + SVM.
        """
        self.device = device
        self.class_names = [f"jzp_{i:03d}" for i in range(1, num_classes + 1)]
        
        # 1. Prepare Backbone (EfficientNet-B0)
        print(f"Loading backbone from: {backbone_path}")
        self.backbone = models.efficientnet_b0(weights=None)
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        try:
            state_dict = torch.load(backbone_path, map_location=self.device)
            self.backbone.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading backbone weights: {e}")
            raise e

        # Remove the final classification layer to use as feature extractor
        self.backbone.classifier = nn.Identity()
        self.backbone.to(self.device).eval()
        
        # 2. Prepare Classifier (SVM)
        print(f"Loading SVM model from: {svm_path}")
        try:
            self.clf = joblib.load(svm_path)
        except Exception as e:
            print(f"Error loading SVM model: {e}")
            raise e
            
        # 3. Preprocessing Pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify_all(self, cropped_images_cv, threshold=0.0):
        """
        Classify a list of cropped OpenCV images.
        """
        predictions = []
        if not cropped_images_cv:
            return []

        input_tensors = []
        for img_cv in cropped_images_cv:
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            input_tensors.append(self.preprocess(img_pil))
        
        batch_tensor = torch.stack(input_tensors).to(self.device)

        with torch.no_grad():
            features = self.backbone(batch_tensor)
            features_np = features.cpu().numpy()

        decision_scores = self.clf.decision_function(features_np)
        pred_indices = self.clf.predict(features_np)

        scores_tensor = torch.tensor(decision_scores)
        probs = F.softmax(scores_tensor, dim=1)
        max_probs, _ = torch.max(probs, dim=1)

        for i, pred_idx in enumerate(pred_indices):
            confidence = max_probs[i].item()
            class_name = self.class_names[pred_idx] if 0 <= pred_idx < len(self.class_names) else "Unknown"

            if confidence < threshold:
                class_name = "Unknown"

            predictions.append((class_name, confidence))
        
        return predictions