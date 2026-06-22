import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    """
    Lightweight Grad-CAM implementation for CNNs.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        # Need requires_grad for backward
        x.requires_grad_(True)
        
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()
            
        score = logits[0, class_idx]
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy() # [C, H, W]
        activations = self.activations[0].cpu().data.numpy() # [C, H, W]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2)) # [C]
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, class_idx, logits[0].softmax(dim=0)[class_idx].item()

def save_gradcam(image_tensor, cam, save_path, true_label=None, pred_label=None, prob=None):
    """
    Overlays CAM on the original image and saves it.
    """
    # Un-normalize image for visualization
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    # Simple denormalization approximation
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.uint8(255 * img)
    
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap) / 255
    img = np.float32(img) / 255
    
    cam_img = heatmap + img
    cam_img = cam_img / np.max(cam_img)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(cam_img)
    plt.axis('off')
    title = f"Pred: {pred_label} ({prob:.2f})"
    if true_label is not None:
        title += f"\nTrue: {true_label}"
    plt.title(title, fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_and_save_gradcam(model, dataset, indices, device, save_dir, prefix="", top_k=3):
    """
    Wrapper to hook model, generate CAM for indices, and save.
    """
    # Auto-detect target layer
    target_layer = None
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'layer4'):
            target_layer = model.backbone.layer4[-1].conv2
    elif hasattr(model, 'features'):
        # For SimpleCNN or similar, pick the last conv layer
        for module in reversed(list(model.features.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
                
    if target_layer is None:
        return # Skip if architecture not supported
        
    cam_extractor = GradCAM(model, target_layer)
    
    for i, idx in enumerate(indices[:top_k]):
        item = dataset[idx]
        if isinstance(item, tuple):
            img_tensor, true_label = item[0], item[1]
        else:
            img_tensor, true_label = item, "Unknown"
            
        x = img_tensor.unsqueeze(0).to(device)
        
        # Fix: Need to ensure model is in train mode or eval but allows gradients
        model.eval()
        for param in model.parameters():
            param.requires_grad = True
            
        try:
            cam, pred_label, prob = cam_extractor(x)
            save_path = os.path.join(save_dir, f"{prefix}_idx{idx}.png")
            save_gradcam(img_tensor, cam, save_path, true_label, pred_label, prob)
        except Exception as e:
            print(f"GradCAM failed for idx {idx}: {e}")
            pass
