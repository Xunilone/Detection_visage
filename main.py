import cv2

# Charger le modèle de détection de visage pré-entraîné
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialiser la capture vidéo depuis la caméra
cap = cv2.VideoCapture(0)

while True:
    # Capturer l'image depuis la caméra
    ret, frame = cap.read()

    # Convertisser l'image en niveaux de gris pour la détection de visage (plus facile pour la détection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détectee les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dessinez un rectangle autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Afficher l'image avec les rectangles de détection
    cv2.imshow('Face Detection', frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Enlever la capture vidéo et fermez la fenêtre d'affichage
cap.release()
cv2.destroyAllWindows()
