# 📌 IA de Génération et d'Évaluation de Modèles 3D

Ce projet utilise **l'intelligence artificielle** pour générer, prétraiter, entraîner et évaluer des **modèles 3D**.  
Il s’appuie sur **PyTorch** pour l'entraînement et **Trimesh** pour la manipulation des objets 3D.

## 🚀 Pipeline IA
L'IA fonctionne en **4 étapes principales** :

### 1️⃣ 🛠️ Génération des modèles 3D  
- Création de modèles 3D aléatoires avec un nombre de sommets contrôlé.  
- Les modèles sont stockés dans le dossier `./data` sous la forme `generated_model_XX.obj`.  

### 2️⃣ 📌 Prétraitement des données  
- Chargement des modèles générés.  
- Nettoyage et transformation pour assurer leur cohérence.  

### 3️⃣ 📈 Entraînement du modèle  
- Utilisation d’un réseau de neurones pour apprendre des modèles 3D.  
- Sauvegarde des poids du modèle (`model.pth`).  

### 4️⃣ 📊 Évaluation du modèle  
- Comparaison des modèles générés avec un modèle de **vérité terrain** (`ground_truth_model.obj`).  
- Rééchantillonnage des sommets pour assurer la correspondance des dimensions.  
- Calcul de **métriques d’erreur** comme **MSE (Mean Squared Error)** et **distance moyenne**.  

---

## 🏗️ Technologies utilisées
- **Python** (3.10+)
- **PyTorch** (Apprentissage profond)
- **Trimesh** (Manipulation de modèles 3D)
- **NumPy** (Traitement des données)

---

## 📂 Structure du projet

```
/data                    # Dossier contenant les fichiers 3D et le modèle entraîné
/scripts                 # Scripts principaux pour chaque étape du pipeline
/models                  # Contient l'architecture du modèle de génération
    model.py             # Définition du réseau de neurones
main.py                  # Lancement du pipeline IA
evaluate.py              # Code pour l'évaluation des modèles
preprocess.py            # Code de prétraitement des modèles
generate.py              # Code pour générer des modèles 3D
train.py                 # Code d'entraînement du modèle
```

## ⚠️ Problèmes connus
- **Erreur `"index -1 is out of bounds for axis 0 with size 0"`**  
  🔹 Problème de rééchantillonnage (résolu par interpolation linéaire).  
- **Les modèles générés sont trop simples**  
  🔹 Ajuster les paramètres de génération.  

---

## 📌 Installation et Exécution

### Sous PowerShell

#### Créer un environnement virtuel python 
*python -m venv venv*

#### Activer l'environnement virtuel
*.venv\Scripts\Activate*

#### Vérifier que l'environnement est bien activé 
*$Env:VIRTUAL_ENV*

#### Installer les dépendances 
*pip install -r requirements.txt*

#### Vérifier les versions installées
*python --version*
*pip freeze | Select-String "torch|trimesh|numpy"*

#### Lancer le projet
*python main.py*

### Sous Bash

#### 1️⃣ Créer un environnement virtuel
*python3 -m venv venv*

#### 2️⃣ Activer l'environnement virtuel
*source venv/bin/activate*

#### 3️⃣ Vérifier que l'environnement est bien activé
*echo $VIRTUAL_ENV*

#### 4️⃣ Installer les dépendances
*pip install -r requirements.txt*

#### 5️⃣ Vérifier les versions installées
*python3 --version*
*pip freeze | grep -E "torch|trimesh|numpy"*

#### 6️⃣ Lancer le projet
*python3 main.py*