import pickle
import trimesh
import torch
import os

def load_pickle(file_path):
    """ Charger un fichier .pkl et afficher son contenu. """
    if not os.path.exists(file_path):
        print(f"❌ Erreur : Le fichier {file_path} est introuvable.")
        return None
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ {file_path} chargé avec succès !")
        print(f"🔍 Type des données : {type(data)}\n")
        return data
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {file_path} : {e}")
        return None

def load_trimesh_model(file_path):
    """ Charger un modèle 3D (STL ou OBJ) avec Trimesh. """
    if not os.path.exists(file_path):
        print(f"❌ Erreur : Le fichier {file_path} est introuvable.")
        return None
    try:
        model = trimesh.load(file_path, force="mesh")
        if isinstance(model, trimesh.Scene):
            geometries = list(model.geometry.values())
            if len(geometries) == 0:
                raise ValueError("La scène ne contient aucune géométrie.")
            model = geometries[0]
        if len(model.vertices) == 0:
            raise ValueError("Le modèle est vide.")
        print(f"✅ Modèle chargé depuis {file_path}")
        print(f"🔍 Nombre de sommets : {len(model.vertices)}")
        print(f"🔍 Nombre de faces : {len(model.faces)}\n")
        for face in model.faces:
            if any(index >= len(model.vertices) or index < 0 for index in face):
                raise ValueError(f"Indices de face invalides dans le modèle : {face}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {file_path} : {e}")
        return None

def validate_obj_model(model, file_path):
    """ Vérifier la validité d’un modèle OBJ en termes de sommets et indices de faces. """
    if model is None:
        return

    if len(model.vertices) < 1000:
        print(f"❌ Erreur : {file_path} ne contient que {len(model.vertices)} sommets (attendu : 1000) !")
    if len(model.faces) == 0:
        print(f"⚠️ Avertissement : {file_path} ne contient aucune face.")

    # Vérifier si des faces contiennent des indices invalides
    for i, face in enumerate(model.faces):
        if any(v >= len(model.vertices) or v < 0 for v in face):
            print(f"❌ Erreur : Indice de face invalide détecté dans {file_path}, face {i} : {face}")
            break
    else:
        print(f"✅ {file_path} est un modèle OBJ valide.\n")

def load_pytorch_model(file_path):
    """ Charger et afficher les informations d’un fichier .pth (modèle PyTorch). """
    if not os.path.exists(file_path):
        print(f"❌ Erreur : Le fichier {file_path} est introuvable.")
        return None
    try:
        # Utilisation sécurisée de torch.load()
        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
        print(f"✅ Modèle PyTorch chargé depuis {file_path}")
        print(f"🔍 Type des données : {type(data)}")
        print(f"🔑 Clés disponibles : {list(data.keys())}\n")
             
        if hasattr(data, "vertices") and len(data.vertices) == 0:
            print("❌ Erreur : `sample_preprocessed.pkl` ne contient aucun sommet !")
        
        return data
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle PyTorch ({file_path}) : {e}\n")
        return None

def open_and_validate_file(filepath):
    try:
        model = trimesh.load(filepath, force="mesh")
        if isinstance(model, trimesh.Scene):
            geometries = list(model.geometry.values())
            if len(geometries) == 0:
                raise ValueError("La scène ne contient aucune géométrie.")
            model = geometries[0]
        if len(model.vertices) == 0:
            raise ValueError("Le modèle est vide.")
        print(f"Nombre de sommets du modèle : {len(model.vertices)}")
        print(f"Nombre de faces du modèle : {len(model.faces)}")
        for face in model.faces:
            if any(index >= len(model.vertices) or index < 0 for index in face):
                raise ValueError(f"Indices de face invalides dans le modèle : {face}")
        return model
    except Exception as e:
        print(f"Erreur lors de l'ouverture et de la validation du fichier : {e}")
        return None

def load_and_display_preprocessed_model(filepath):
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        if len(model.vertices) == 0:
            raise ValueError("Le modèle prétraité est vide.")
        print(f"Nombre de sommets du modèle prétraité : {len(model.vertices)}")
        print(f"Nombre de faces du modèle prétraité : {len(model.faces)}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle prétraité : {e}")
        return None

def main():
    """ Fonction principale pour charger et vérifier tous les fichiers. """
    print("\n=== 📂 Chargement des fichiers et validation ===\n")

    file_path_pkl = "./data/sample_preprocessed.pkl"
    file_path_stl = "./data/sample.stl"
    file_path_pth = "./data/model.pth"
    file_path_obj = "./data/ground_truth_model.obj"

    # Charger les fichiers
    load_pickle(file_path_pkl)
    model_stl = load_trimesh_model(file_path_stl)
    model_obj = load_trimesh_model(file_path_obj)
    validate_obj_model(model_obj, file_path_obj)
    load_pytorch_model(file_path_pth)

    print("\n=== ✅ Vérifications terminées ===")

if __name__ == "__main__":
    main()
    filepath = "./data/ground_truth_model.obj"
    model = open_and_validate_file(filepath)
    if model:
        print("Fichier ouvert et validé avec succès.")
    else:
        print("Échec de l'ouverture et de la validation du fichier.")
    
    preprocessed_filepath = "./data/sample_preprocessed.pkl"
    preprocessed_model = load_and_display_preprocessed_model(preprocessed_filepath)
    if preprocessed_model:
        print("Modèle prétraité chargé et affiché avec succès.")
    else:
        print("Échec du chargement et de l'affichage du modèle prétraité.")
