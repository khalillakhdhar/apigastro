import torch

MODEL_PATH = "best_vqa_model.pth"

# Charger le fichier
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    print("\n✅ Fichier chargé avec succès. Voici les premières clés :\n")
    for i, key in enumerate(state_dict.keys()):
        print(f"{i+1:03d}: {key}")
        if i >= 50:
            print("\n... (affichage limité à 50 clés)")
            break
else:
    print(" Le fichier ne contient pas 'model_state_dict'. Voici les premières clés disponibles :")
    for i, key in enumerate(checkpoint.keys()):
        print(f"{i+1:03d}: {key}")