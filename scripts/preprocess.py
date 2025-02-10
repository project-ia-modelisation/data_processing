import numpy as np
import trimesh
import os
import pickle

def preprocess_model(model, num_vertices=1000):
    """
    Prétraitement : normalisation et réduction de résolution.
    """
    # Convertir le modèle en numpy array si ce n'est pas déjà fait
    vertices = np.array(model.vertices)
    
    # Normalisation
    center = np.mean(vertices, axis=0)
    vertices = vertices - center  # Centrer le modèle
    
    max_extent = np.max(np.abs(vertices))
    if max_extent == 0:
        raise ValueError("Le modèle a des dimensions nulles, impossible de normaliser.")
    
    vertices = vertices / max_extent  # Normaliser entre -1 et 1
    
    # Mettre à jour le modèle
    model.vertices = vertices
    
    # Rééchantillonnage
    if len(model.vertices) > num_vertices:
        model = model.simplify_quadratic_decimation(num_vertices)
    elif len(model.vertices) < num_vertices:
        points = model.sample(num_vertices)
        model = trimesh.Trimesh(vertices=points)
    
    return model

def load_and_preprocess_model(file_path, num_vertices=1000):
    """
    Charger un modèle 3D et appliquer un prétraitement.
    
    Args:
        file_path (str): Chemin du fichier contenant le modèle 3D.
        num_vertices (int): Nombre de sommets à rééchantillonner.
        
    Returns:
        trimesh.Trimesh: Modèle 3D prétraité.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier spécifié n'existe pas : {file_path}")
    
    try:
        # Charger le modèle
        model = trimesh.load_mesh(file_path)
        
        # S'assurer que c'est un Trimesh
        if isinstance(model, trimesh.Scene):
            geometries = list(model.geometry.values())
            if len(geometries) == 0:
                raise ValueError("La scène ne contient aucune géométrie.")
            model = geometries[0]
        
        # Vérifier si le modèle contient des sommets et des faces
        if len(model.vertices) == 0 or len(model.faces) == 0:
            raise ValueError("Le modèle chargé est vide.")
        
        # Prétraitement
        preprocessed_model = preprocess_model(model, num_vertices)
        return preprocessed_model
        
    except Exception as e:
        raise Exception(f"Erreur lors du chargement/prétraitement du modèle : {e}")

def load_preprocessed_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    if len(model.vertices) == 0:
        raise ValueError("Le modèle prétraité est vide.")
    return model