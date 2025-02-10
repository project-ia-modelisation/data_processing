import numpy as np

# Définition du fichier de sortie
output_path = "./data/ground_truth_model.obj"

# Générer exactement 1000 sommets dans un espace [-1, 1]
num_vertices = 1500
vertices = np.random.uniform(-1, 1, (num_vertices, 3))

# Générer 2000 faces avec des indices valides (sans dépasser 1000)
num_faces = 2000
faces = np.random.randint(1, num_vertices + 1, (num_faces, 3))

# Écrire dans le fichier OBJ
with open(output_path, "w") as f:
    f.write("# Ground Truth Model\n")
    f.write("# 1000 sommets et 2000 faces\n\n")

    # Écrire les sommets
    for v in vertices:
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

    f.write("\n# Définition des faces\n")
    
    # Écrire les faces
    for face in faces:
        f.write(f"f {face[0]} {face[1]} {face[2]}\n")

print(f"✅ Nouveau modèle généré dans {output_path} avec succès !")
