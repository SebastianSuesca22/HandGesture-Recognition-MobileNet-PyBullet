import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pybullet as p
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class HandGestureRecognition:
    def __init__(self):
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Configurar MobileNet
        self.img_height, self.img_width = 224, 224
        self.create_mobilenet_model()

        # Inicializar PyBullet
        self.client = p.connect(p.GUI)  # Inicia en modo gráfico
        p.setGravity(0, 0, -9.81)  # Gravedad
        p.setTimeStep(1. / 240.)  # Tiempo de simulación

        # Crear la "mesa" (una banda de varios cubos)
        self.belt_width = 5
        self.belt_length = 10
        self.belt_height = 0.1  # Altura de la banda (mesa)
        self.belt_position = [0, 0, -0.5]  # Colocamos la banda un poco debajo de la esfera

        # Crear cubos que simulan la banda transportadora
        self.cube_size = 1  # Tamaño de cada cubo
        self.cubes = []
        for i in range(int(self.belt_length / self.cube_size)):
            cube_position = [i * self.cube_size - self.belt_length / 2, 0, -0.5]
            cube_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.cube_size / 2, self.belt_width / 2, self.belt_height / 2]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[self.cube_size / 2, self.belt_width / 2, self.belt_height / 2]),
                basePosition=cube_position
            )
            self.cubes.append(cube_id)

        # Crear 5 esferas más pequeñas, distribuidas más alejadas
        self.sphere_radius = 0.25  # Esfera más pequeña
        self.sphere_start_pos = [[-4, 0, 0.25], [-2.5, 0, 0.25], [0, 0, 0.25], [2.5, 0, 0.25], [4, 0, 0.25]]  # Posiciones distribuidas más alejadas
        self.sphere_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # Crear 5 esferas con posiciones distribuidas
        self.spheres = []
        self.spheres_color = [(1, 0, 0)] * 5  # Color inicial rojo
        self.spheres_state = [False] * 5  # Estado de las esferas (False = apagada, True = encendida)
        for pos in self.sphere_start_pos:
            sphere_id = p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.sphere_radius),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=self.sphere_radius),
                basePosition=pos,
                baseOrientation=self.sphere_start_orientation
            )
            self.spheres.append(sphere_id)

        # Variables de control de las esferas
        self.spheres_pos = self.sphere_start_pos  # Lista con las posiciones de las 5 esferas

    def create_mobilenet_model(self):
        """Crear modelo personalizado con MobileNet"""
        base_model = MobileNet(
            weights='imagenet', 
            include_top=False, 
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(6, activation='softmax', name='gesture_output')(x)  # 0-5 gestos
        
        self.model = Model(inputs=base_model.input, outputs=output)
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def count_fingers(self, hand_landmarks):
        """Método de conteo de dedos más simple y directo"""
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_mcp = [
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        
        extended_fingers = 0
        for i in range(1, 5):  # Omitir pulgar inicialmente
            tip = hand_landmarks.landmark[finger_tips[i]]
            base = hand_landmarks.landmark[finger_mcp[i]]
            if tip.y < base.y:
                extended_fingers += 1
        
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        is_right_hand = wrist.x > thumb_mcp.x
        thumb_extended = (
            (is_right_hand and thumb_tip.x < thumb_mcp.x) or 
            (not is_right_hand and thumb_tip.x > thumb_mcp.x)
        )
        
        total_fingers = extended_fingers + (1 if thumb_extended else 0)
        
        return min(total_fingers, 5)

    def detect_and_classify_hand(self, frame):
        """Detectar mano y clasificar gesto"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        num_fingers = 0
        gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                num_fingers = self.count_fingers(hand_landmarks)
                
                hand_img = self.extract_hand_region(frame, hand_landmarks)
                
                if hand_img is not None:
                    preprocessed_img = self.preprocess_hand_image(hand_img)
                    prediction = self.model.predict(preprocessed_img)
                    gesture = num_fingers
                    
                    # Cambiar el color de las esferas según el gesto
                    self.change_sphere_color(gesture)

        return num_fingers, gesture

    def preprocess_hand_image(self, hand_img):
        """Preprocesar imagen de la mano para MobileNet"""
        img = cv2.resize(hand_img, (self.img_width, self.img_height))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def extract_hand_region(self, frame, hand_landmarks):
        """Extraer región de la mano"""
        x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
        x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
        y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
        y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
        
        padding = 20
        x_min = max(0, int(x_min - padding))
        x_max = min(frame.shape[1], int(x_max + padding))
        y_min = max(0, int(y_min - padding))
        y_max = min(frame.shape[0], int(y_max + padding))
        
        hand_img = frame[y_min:y_max, x_min:x_max]
        
        return hand_img

    def change_sphere_color(self, gesture):
        """Cambiar color de las esferas basadas en el gesto"""
        # Apagar todas las esferas si se detectan 5 dedos
        if gesture == 5:
            self.spheres_state = [False] * 5
            self.spheres_color = [(1, 0, 0)] * 5  # Todas las esferas se ponen rojas (apagadas)
        elif 1 <= gesture <= 4:
            self.spheres_state[gesture - 1] = True
            self.spheres_color = [(0, 1, 0) if state else (1, 0, 0) for state in self.spheres_state]  # Verde para la esfera activada, roja para las demás
        
        # Actualizar el color de las esferas en PyBullet
        for i in range(5):
            p.changeVisualShape(self.spheres[i], -1, rgbaColor=self.spheres_color[i] + (1,))

def main():
    cap = cv2.VideoCapture(0)
    hand_gesture = HandGestureRecognition()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        num_fingers, gesture = hand_gesture.detect_and_classify_hand(frame)
        
        info_text = f"Dedos: {num_fingers}, Gesto: {gesture}"
        cv2.putText(frame, info_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 0), 2)
        
        cv2.imshow('Reconocimiento de Gestos', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()  # Desconectar la simulación de PyBullet

if __name__ == "__main__":
    main()