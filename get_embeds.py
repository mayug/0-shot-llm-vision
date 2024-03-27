import torchvision.datasets as dset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel, ViTConfig, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, models
from transformers import AutoImageProcessor, AutoModel
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from transformers import AutoImageProcessor, DetrModel
from transformers import SegformerForSemanticSegmentation
from PIL import Image
import requests
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import SamModel, SamProcessor
from transformers import SegformerModel
from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import timm
import numpy as np


import torch, random
from tqdm import tqdm
import sys
import re

COCO_ROOT = "/datasets/coco_2024-01-04_1601/val2017"
NOCAPS_ROOT = "/shared/group/openimages/validation"
COCO_ANN = "/datasets/coco_2024-01-04_1601/annotations/captions_val2017.json"
NOCAPS_ANN = "nocaps_val_4500_captions.json"

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--m",
        dest="model_name",
        help="model_name",
        default="dinov2",
        type=str,
    )
    parser.add_argument(
        "--d",
        dest="dataset",
        help="dataset",
        default="coco",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        help="gpu",
        default=7,
        type=int,
    )
    return parser.parse_args()

class FeatureExtractor:
    def __init__(self):
        self.extracted_features = None

    def __call__(self, module, input_, output):
        self.extracted_features = output

def get_model(model_name, device):
    
    if model_name == "convnext":
        processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224-22k")
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224-22k").to(device)
        return model, processor
    elif model_name == "dinov2":
        vision_model_name = "facebook/dinov2-large"
        processor = AutoImageProcessor.from_pretrained(vision_model_name)
        model = AutoModel.from_pretrained(vision_model_name).to(device)
        return model, processor
    elif model_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        return model, processor
    elif model_name == "allroberta":
        language_model_name = "all-roberta-large-v1"
        language_model = SentenceTransformer(language_model_name).to(device)
        return language_model
    elif model_name == "detr_resnet_50_encoder":
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrModel.from_pretrained("facebook/detr-resnet-50").to(device)
        
        extractor = FeatureExtractor()
        model.encoder.register_forward_hook(extractor)
        return model
    elif model_name == "detr_resnet_50_decoder":
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrModel.from_pretrained("facebook/detr-resnet-50").to(device)
        
        extractor = FeatureExtractor()
        model.decoder.register_forward_hook(extractor)
        return model
    elif model_name == "detr_resnet_50_backbone":
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrModel.from_pretrained("facebook/detr-resnet-50").to(device)
        
        extractor = FeatureExtractor()
        model.backbone.conv_encoder.register_forward_hook(extractor)
        return model
    elif model_name == "detr_resnet_101_backbone":
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-101")
        model = DetrModel.from_pretrained("facebook/detr-resnet-101").to(device)
        
        extractor = FeatureExtractor()
        model.backbone.conv_encoder.register_forward_hook(extractor)
        return model
    elif model_name == "detr_resnet_101_encoder":
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-101")
        model = DetrModel.from_pretrained("facebook/detr-resnet-101").to(device)
        
        extractor = FeatureExtractor()
        model.encoder.register_forward_hook(extractor)
        return model
    elif model_name == "detr_resnet_101_decoder":
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-101")
        model = DetrModel.from_pretrained("facebook/detr-resnet-101").to(device)
        
        extractor = FeatureExtractor()
        model.decoder.register_forward_hook(extractor)
        return model
    elif model_name == "sam":
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
        extractor = FeatureExtractor()
        model.vision_encoder.layers[31].register_forward_hook(extractor)
        return model
    elif model_name == "sam_embed":
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
        extractor = FeatureExtractor()
        model.shared_image_embedding.register_forward_hook(extractor)
        return model
    elif model_name == "segformer":
        processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
        model = SegformerModel.from_pretrained("nvidia/mit-b0").to(device)

        extractor = FeatureExtractor()
        model.encoder.register_forward_hook(extractor)
        return model
    elif model_name == "segformer_segment":
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)

        extractor = FeatureExtractor()
        model.segformer.encoder.register_forward_hook(extractor)
        return model
    elif model_name == "dpt":
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

        extractor = FeatureExtractor()
        model.dpt.encoder.register_forward_hook(extractor)
        return model
    elif model_name == "resnet101":
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-101")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-101").to(device)

        extractor = FeatureExtractor()
        model.resnet.encoder.register_forward_hook(extractor)
        return model
    elif model_name == "vit":
        model = timm.create_model(
            'vit_base_patch16_384.augreg_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).to(device)
        model = model.eval()
        
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return model, transform

def get_dataset(dataset):
    if dataset=="coco":
        cap = dset.CocoCaptions(root = COCO_ROOT,
                        annFile = COCO_ANN)#,
                        #transform=transforms.Compose([
                            # transforms.Resize((256,256)), 
                            # transforms.RandomResizedCrop((224,224)), 
                            #transforms.PILToTensor()]),)
                            # target_transform=select_first_k_captions)
    elif dataset=="nocaps":
        cap = dset.CocoCaptions(root = NOCAPS_ROOT,
                        annFile = NOCAPS_ANN,
                        transform=transforms.Compose([
                            #transforms.Resize((256,256)), 
                            #transforms.RandomResizedCrop((224,224)), 
                            transforms.PILToTensor()]),)
    return cap 

def run_model(model_name, model_transform, cap, device):
    if model_name == "dinov2":
        model, processor = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):          
            inputs = processor(images=img, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            image_representation = outputs.last_hidden_state.mean(dim=1).detach().cpu()[0]
            image_representations.append(image_representation)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif "resnet101" == model_name:
        model = model_transform
        image_representations = []
        for img, target in tqdm(cap):
            inputs = feature_extractor(img, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            output = extractor.extracted_features.last_hidden_state.reshape((1, 2048, -1)).mean(dim=-1).detach().cpu()[0]
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif "detr_resnet_50_backbone" == model_name or "detr_resnet_101_backbone" == model_name:
        model = model_transform
        image_representations = []
        for img, target in tqdm(cap):
            inputs = image_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            output = extractor.extracted_features[-1][0].reshape((1, 2048, -1)).mean(dim=-1).detach().cpu()[0]
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif "detr_resnet" in model_name:
        model = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            inputs = image_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            output = extractor.extracted_features.last_hidden_state.mean(dim=1).detach().cpu()[0]
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "vit":
        model, transform = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            input = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input)[0]  # output is (batch_size, num_features) shaped tensor
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "sam":
        model = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            output = extractor.extracted_features[0].reshape(1, -1, 1280).mean(dim = 1).detach().cpu()[0]
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "sam_embed":
        model = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            output = extractor.extracted_features.reshape(-1, 256).mean(dim = 0).detach().cpu()
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "segformer" or model_name == "segformer_segment":
        model = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            output = extractor.extracted_features.last_hidden_state.reshape((1, 256, -1)).mean(-1).detach().cpu()[0]
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "dpt":
        model = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            output = extractor.extracted_features.last_hidden_state.mean(dim=1).detach().cpu()[0]
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "convnext":
        model = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            
            inputs = processor(images=img, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model.convnext(**inputs)
            #print(outputs.last_hidden_state.shape)
            image_representation = outputs.last_hidden_state.reshape(1, 1024, -1).mean(2).detach().cpu()[0]
            image_representation = image_representation / np.linalg.norm(image_representation, axis=0, keepdims=True)
            image_representations.append(image_representation)

        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "clip":
        model, processor = model_transform
        image_representations = []
        text_representations = []
        
        for img, target in tqdm(cap):
            
            inputs = processor(text=target, images=img, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            outputs = model(**inputs)
            text_representation = outputs.text_embeds.detach().cpu().squeeze()
            text_representation = text_representation.mean(dim=0).squeeze()
            image_representation = outputs.image_embeds.detach().cpu().squeeze()
        
            text_representations.append(text_representation)
            image_representations.append(image_representation)

        text_representations_tensor = torch.stack(text_representations)
        image_representations_tensor = torch.stack(image_representations)

        torch.save(text_representations_tensor, f'data/{dataset}_{model_name}_text.pt')
        torch.save(image_representations_tensor, f'data/{dataset}_{model_name}_img.pt')
    elif model_name == "allroberta":
        language_model = model_transform
        text_representations = []

        for img, target in tqdm(cap):
            output = language_model.encode(target)
            text_representation = torch.Tensor(output)
            text_representation = text_representation.mean(dim=0)
            text_representations.append(text_representation)
            
        text_representations_tensor = torch.stack(text_representations)
        torch.save(text_representations_tensor, f'data/{dataset}_{model_name}_text.pt')

if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    gpu = 'cuda'
    device = torch.device(gpu)

    model_name = args.model_name
    dataset = args.dataset

    model_transform = get_model(model_name, device)
    cap = get_model(dataset)
    run_model(model_name, model_transform, cap, device)

    

    
    
    

    
        
    



