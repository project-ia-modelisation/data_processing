from scripts.preprocess import load_preprocessed_model
from scripts.evaluate import evaluate_model, resample_vertices
from models.model import Simple3DGenerator
from scripts.generate import generate_and_save_model
from scripts.train import train_model
from datetime import datetime
import os
import time
import torch
import trimesh
import numpy as np

def correct_invalid_faces(model):
    """
    Vérifie et corrige les faces contenant des indices invalides.
    """
    max_index = len(model.vertices) - 1
    valid_faces = [face for face in model.faces if all(0 <= v <= max_index for v in face)]
    model.faces = np.array(valid_faces)
    print(f"✅ Modèle corrigé : {len(model.faces)} faces après suppression des indices invalides.")

def generate_shapes(modele, dossier_sortie="./data"):
    try:
        print("\n[Étape 1] Génération de nouvelles formes...")
        generate_and_save_model(modele, output_dir=dossier_sortie, min_vertices=100, max_vertices=1000)
        print("Génération des formes terminée avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de la génération des formes : {e}")

def preprocess_data():
    try:
        print("\n[Étape 2] Prétraitement des données...")
        # Ajoutez ici le code de prétraitement si nécessaire
        print("✅ Prétraitement terminé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du prétraitement des données : {e}")

def train_model_on_generated_data():
    try:
        print("\n[Étape 3] Entraînement du modèle...")
        fichiers_modeles = [os.path.join("./data", f) for f in os.listdir("./data") if f.startswith("generated_model_") and f.endswith(".obj")]
        if not fichiers_modeles:
            print("⚠️ Aucun fichier généré trouvé pour l'entraînement.")
            return
        train_model(fichiers_modeles)
        print("✅ Entraînement terminé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du modèle : {e}")

def evaluate_model_on_ground_truth():
    print("\n[Étape 4] Évaluation du modèle...")

    try:
        print("📂 Chargement du modèle prétraité...")
        preprocessed_model = load_preprocessed_model("./data/sample_preprocessed.pkl")
        
        if not isinstance(preprocessed_model, trimesh.Trimesh):
            raise ValueError(f"❌ ERREUR : `preprocessed_model` n'est pas un Trimesh mais {type(preprocessed_model)}")
        
        print(f"✅ Modèle prétraité chargé : {type(preprocessed_model)} avec {len(preprocessed_model.vertices)} sommets.")

        ground_truth_files = [os.path.join("./data", f) for f in os.listdir("./data") if f.endswith(".obj") and not f.startswith("generated_model_")]
        if not ground_truth_files:
            print("⚠️ Aucun fichier de vérité terrain trouvé pour l'évaluation.")
            return

        for ground_truth_file in ground_truth_files:
            print(f"📂 Chargement du modèle vérité terrain : {ground_truth_file}")
            try:
                ground_truth_model = trimesh.load(ground_truth_file, force="mesh")

                if not isinstance(ground_truth_model, trimesh.Trimesh):
                    raise ValueError(f"❌ ERREUR : `ground_truth_model` n'est pas un Trimesh mais {type(ground_truth_model)}")

                print(f"✅ Modèle vérité terrain chargé : {type(ground_truth_model)} avec {len(ground_truth_model.vertices)} sommets.")

                # 🛑 Vérification des tailles avant rééchantillonnage
                print(f"🔍 Sommets avant rééchantillonnage (prétraité) : {len(preprocessed_model.vertices)}")
                print(f"🔍 Sommets avant rééchantillonnage (vérité terrain) : {len(ground_truth_model.vertices)}")

                if len(preprocessed_model.vertices) == 0 or len(ground_truth_model.vertices) == 0:
                    raise ValueError("❌ Erreur critique : Un des modèles est VIDE avant rééchantillonnage !")

                # 🔄 Rééchantillonnage des sommets
                preprocessed_model_resampled = resample_vertices(preprocessed_model.vertices, len(ground_truth_model.vertices))
                ground_truth_vertices_resampled = resample_vertices(ground_truth_model.vertices, len(ground_truth_model.vertices))

                print(f"📊 Sommets après rééchantillonnage (prétraité) : {len(preprocessed_model_resampled)}")
                print(f"📊 Sommets après rééchantillonnage (vérité terrain) : {len(ground_truth_vertices_resampled)}")

                if len(preprocessed_model_resampled) == 0 or len(ground_truth_vertices_resampled) == 0:
                    raise ValueError("❌ Erreur critique : Un des modèles est VIDE après rééchantillonnage !")

                # 🚨 Vérification avant conversion en Trimesh
                print("🔄 Conversion des sommets rééchantillonnés en Trimesh...")
                try:
                    preprocessed_model = trimesh.Trimesh(vertices=preprocessed_model_resampled, faces=preprocessed_model.faces)
                    ground_truth_model = trimesh.Trimesh(vertices=ground_truth_vertices_resampled, faces=ground_truth_model.faces)
                except Exception as e:
                    raise ValueError(f"❌ ERREUR lors de la conversion en Trimesh : {e}")

                print("✅ Conversion en Trimesh réussie.")

                # 🔍 Vérification finale avant évaluation
                if not isinstance(preprocessed_model, trimesh.Trimesh):
                    raise ValueError(f"❌ ERREUR : `preprocessed_model` final n'est pas un Trimesh mais {type(preprocessed_model)}")

                if not isinstance(ground_truth_model, trimesh.Trimesh):
                    raise ValueError(f"❌ ERREUR : `ground_truth_model` final n'est pas un Trimesh mais {type(ground_truth_model)}")

                # 🔄 Lancer l'évaluation
                metrics = evaluate_model(preprocessed_model, ground_truth_model)
                print(f"✅ Évaluation terminée avec succès. 📊 Résultats des métriques : {metrics}")

            except Exception as e:
                print(f"❌ Erreur lors de l'évaluation avec {ground_truth_file} : {e}")

    except ValueError as ve:
        print(f"❌ Erreur de validation : {str(ve)}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {str(e)}")

def main():
    print("=== 🚀 Lancement du pipeline IA ===")

    if not os.path.exists("./data/model.pth"):
        print("❌ Erreur : Les poids du modèle n'ont pas été trouvés.")
        return

    try:
        while True:
            # Initialisation du modèle
            modele = Simple3DGenerator()
            modele.load_state_dict(torch.load("./data/model.pth", map_location=torch.device('cpu'), weights_only=True))
            modele.eval()

            print(f"\n[{datetime.now()}] 🔄 Début d'une nouvelle itération du pipeline.")
            generate_shapes(modele)
            preprocess_data()
            train_model_on_generated_data()
            evaluate_model_on_ground_truth()
            print(f"[{datetime.now()}] ✅ Fin de l'itération. Pause de 120 secondes.")
            time.sleep(120)

            if os.path.exists("./stop_pipeline.flag"):
                print("🚨 Fichier de signal trouvé. Arrêt propre.")
                break
    except KeyboardInterrupt:
        print("🛑 Interruption du programme. Arrêt propre.")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")

if __name__ == "__main__":
    main()
