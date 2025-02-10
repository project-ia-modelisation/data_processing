import pickle
import trimesh
from stl import mesh
import numpy as np
import open3d as o3d
import os
import cv2  # Add OpenCV import

# Lecture d'un fichier pkl
def read_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"Lecture du fichier pickle réussie : {file_path}")
        return data
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier pickle : {file_path}, Erreur : {e}")
        return None

# Écriture d'un fichier pkl
def write_pkl(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Écriture du fichier pickle réussie : {file_path}")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier pickle : {file_path}, Erreur : {e}")

# Lecture d'un fichier STL avec trimesh
def read_stl_trimesh(file_path):
    try:
        mesh = trimesh.load(file_path)
        print(f"Lecture du fichier STL avec trimesh réussie : {file_path}")
        return mesh
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier STL avec trimesh : {file_path}, Erreur : {e}")
        return None

# Sauvegarde en STL avec trimesh
def save_stl_trimesh(mesh, file_path):
    try:
        mesh.export(file_path)
        print(f"Sauvegarde du fichier STL avec trimesh réussie : {file_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier STL avec trimesh : {file_path}, Erreur : {e}")

# Lecture d'un fichier STL avec numpy-stl
def read_stl_numpy(file_path):
    try:
        mesh_data = mesh.Mesh.from_file(file_path)
        print(f"Lecture du fichier STL avec numpy-stl réussie : {file_path}")
        return mesh_data
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier STL avec numpy-stl : {file_path}, Erreur : {e}")
        return None

# Lecture d'un fichier image 2D
def read_image(file_path):
    try:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Erreur lors de la lecture de l'image")
        print(f"Lecture de l'image réussie : {file_path}")
        return image
    except Exception as e:
        print(f"Erreur lors de la lecture de l'image : {file_path}, Erreur : {e}")
        return None

# Sauvegarde d'un fichier image 2D
def save_image(image, file_path):
    try:
        cv2.imwrite(file_path, image)
        print(f"Sauvegarde de l'image réussie : {file_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image : {file_path}, Erreur : {e}")

# Évaluation d'une image 2D
def evaluate_image(generated_image, ground_truth_image):
    try:
        if generated_image.shape != ground_truth_image.shape:
            raise ValueError("Les images doivent avoir la même forme pour l'évaluation.")
        
        mse = np.mean((generated_image - ground_truth_image) ** 2)
        max_error = np.max(np.abs(generated_image - ground_truth_image))
        average_distance = np.mean(np.linalg.norm(generated_image - ground_truth_image, axis=2))
        
        metrics = {
            "mean_squared_error": mse,
            "max_error": max_error,
            "average_distance": average_distance
        }
        return metrics
    except Exception as e:
        print(f"Erreur lors de l'évaluation de l'image : {e}")
        return None

def load_and_validate_model(filepath):
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
        print(f"Erreur lors du chargement et de la validation du modèle : {e}")
        return None

if __name__ == "__main__":
    # Exemple d'utilisation des fonctions avec des fichiers dans le dossier data
    data_dir = "./data"
    
    # Lecture d'un fichier pickle
    pkl_file_path = os.path.join(data_dir, "sample_preprocessed.pkl")
    pkl_data = read_pkl(pkl_file_path)
    
    # Écriture d'un fichier pickle
    new_pkl_file_path = os.path.join(data_dir, "new_sample_preprocessed.pkl")
    write_pkl(pkl_data, new_pkl_file_path)
    
    # Lecture d'un fichier STL avec trimesh
    stl_file_path = os.path.join(data_dir, "sample.stl")
    stl_mesh = read_stl_trimesh(stl_file_path)
    
    # Sauvegarde d'un fichier STL avec trimesh
    new_stl_file_path = os.path.join(data_dir, "new_sample.stl")
    save_stl_trimesh(stl_mesh, new_stl_file_path)
    
    # Lecture d'un fichier STL avec numpy-stl
    stl_mesh_numpy = read_stl_numpy(stl_file_path)
    
    # Lecture d'une image 2D
    image_file_path = os.path.join(data_dir, "sample_image.png")
    image_data = read_image(image_file_path)
    
    # Sauvegarde d'une image 2D
    new_image_file_path = os.path.join(data_dir, "new_sample_image.png")
    save_image(image_data, new_image_file_path)

    # Exemple d'utilisation
    filepath = "./data/ground_truth_model.obj"
    model = load_and_validate_model(filepath)
    if model:
        print("Modèle chargé et validé avec succès.")
    else:
        print("Échec du chargement et de la validation du modèle.")