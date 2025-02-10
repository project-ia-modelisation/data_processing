import open3d as o3d
import numpy as np
import os

def visualize_model(file_path, window_name="Visualisation", output_dir=None, epoch=None):
    """
    Visualise un modèle 3D à partir d'un fichier .obj
    """
    # Charger le modèle
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Créer une fenêtre de visualisation
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # Ajouter le mesh
    vis.add_geometry(mesh)
    
    # Configurer la vue
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.2, 0.2, 0.2])  # Fond gris foncé
    opt.point_size = 1.0
    
    # Lancer la visualisation
    vis.run()
    if output_dir is not None and epoch is not None:
        vis.capture_screen_image(f"{output_dir}/epoch_{epoch}.png")
    vis.destroy_window()

def visualize_all_models_in_directory(directory):
    """
    Visualise tous les fichiers .obj dans un répertoire donné
    """
    geometries = []
    for filename in os.listdir(directory):
        if filename.endswith(".obj"):
            file_path = os.path.join(directory, filename)
            print(f"Visualisation du fichier : {file_path}")
            try:
                mesh = o3d.io.read_triangle_mesh(file_path)
                geometries.append(mesh)
            except Exception as e:
                print(f"Erreur lors de la visualisation du fichier {file_path} : {e}")
    
    if geometries:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Visualisation de tous les modèles")
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.run()
        vis.destroy_window()

# Utilisation
visualize_all_models_in_directory("./data")

def monitor_training_progress(output_dir, epoch, model_path):
    """
    Visualise le modèle après chaque N époques
    """
    try:
        visualize_model(model_path, f"Epoch {epoch}", output_dir, epoch)
    except Exception as e:
        print(f"Erreur lors de la visualisation : {e}")