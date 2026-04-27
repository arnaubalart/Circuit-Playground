import cv2
import numpy as np
import time
import math

def draw_led(canvas, x, y, is_on):
    """YELLOW LED"""
    if is_on:
        cv2.circle(canvas, (x, y), 30, (0, 50, 50), -1)   
        cv2.circle(canvas, (x, y), 20, (0, 150, 150), -1)  
        cv2.circle(canvas, (x, y), 8, (150, 255, 255), -1) 
    else:
        cv2.circle(canvas, (x, y), 12, (50, 50, 50), 2)
        cv2.circle(canvas, (x, y), 8, (20, 20, 20), -1)

def draw_switch(canvas, x, y, is_on):
    # GREEN or RED HALO SWITCH
    overlay = canvas.copy()
    

    if is_on:

        cv2.circle(overlay, (x, y), 45, (0, 255, 0), -1) 
    else:

        cv2.circle(overlay, (x, y), 45, (0, 0, 255), -1)
        
    cv2.addWeighted(overlay, 0.3, canvas, 1.0, 0, canvas)
    
    if is_on:
        cv2.circle(canvas, (x, y), 10, (100, 255, 100), -1) 
    else:
        cv2.circle(canvas, (x, y), 10, (100, 100, 100), -1) 

def draw_oscilloscope(canvas, x_pos, y_pos, current_time, voltage_amp, current_amp):
    """
    Dibuja un panel flotante estilo osciloscopio mostrando las ondas de voltaje y corriente.
    x_pos, y_pos: Coordenadas de la esquina superior izquierda del panel.
    """
    width, height = 200, 100
    center_y = y_pos + height // 2
    
    # 1. Dibujar el fondo del panel (Gris muy oscuro y semitransparente)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + width, y_pos + height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, canvas, 1.0, 0, canvas)
    
    # 2. Dibujar el marco y la línea central (eje cero)
    cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + width, y_pos + height), (100, 100, 100), 1)
    cv2.line(canvas, (x_pos, center_y), (x_pos + width, center_y), (50, 50, 50), 1)
    
    # 3. Calcular los puntos de la onda
    pts_voltage = []
    pts_current = []
    
    # Ajustes visuales de la onda
    freq = 0.05   # Cómo de comprimidas están las ondas (frecuencia)
    speed = 5.0   # Velocidad a la que se desplazan visualmente
    
    # Iteramos a lo largo del ancho del panel (200 píxeles)
    for x_offset in range(width):
        # El desplazamiento en el tiempo crea el efecto de movimiento
        t = (x_offset * freq) - (current_time * speed)
        
        # Eje Y del Voltaje (Onda Amarilla)
        # Multiplicamos por la amplitud. Restamos en lugar de sumar porque en OpenCV la Y va hacia abajo
        v_y = int(center_y - (math.sin(t) * voltage_amp))
        pts_voltage.append([x_pos + x_offset, v_y])
        
        # Eje Y de la Corriente (Onda Azul)
        # Podemos añadir un desfase sumando a 't' si la carga es inductiva/capacitiva, aquí lo hacemos en fase
        i_y = int(center_y - (math.sin(t) * current_amp))
        pts_current.append([x_pos + x_offset, i_y])
        
    # 4. Convertir listas a matrices de NumPy preparadas para OpenCV
    # La forma (-1, 1, 2) es un requisito estricto de la función cv2.polylines
    array_v = np.array(pts_voltage, np.int32).reshape((-1, 1, 2))
    array_c = np.array(pts_current, np.int32).reshape((-1, 1, 2))
    
    # 5. Dibujar las líneas de las ondas
    cv2.polylines(canvas, [array_v], isClosed=False, color=(0, 255, 255), thickness=2) # Voltaje (Amarillo)
    cv2.polylines(canvas, [array_c], isClosed=False, color=(255, 150, 0), thickness=2) # Corriente (Azul claro)
    
    # 6. Añadir las leyendas
    cv2.putText(canvas, "V", (x_pos + 5, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(canvas, "I", (x_pos + 5, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

def animate_wire(canvas, x1, y1, x2, y2, current_time, speed, is_on):
    """Animates 'electrons' traveling from one point to another."""
    # 1. Draw the base wire (faint so it doesn't distract)
    cv2.line(canvas, (x1, y1), (x2, y2), (30, 30, 30), 2)
    
    if not is_on:
        return # If off, no electrons are moving
        
    # 2. Math to move the points
    distance = math.hypot(x2 - x1, y2 - y1)
    num_electrons = int(distance / 30) # One electron every 30 pixels
    
    for i in range(num_electrons):
        # The modulo % 1.0 makes the value loop between 0.0 and 1.0
        progress = ((current_time * speed) + (i / num_electrons)) % 1.0
        
        # Linear interpolation: calculate exact X and Y for this frame
        ex = int(x1 + (x2 - x1) * progress)
        ey = int(y1 + (y2 - y1) * progress)
        
        # Draw the electron (bright yellow)
        cv2.circle(canvas, (ex, ey), 3, (255, 255, 0), -1)

def main():
    print("--- VFX ENGINE STARTED ---")
    print("Controls:")
    print("[SPACE] : Turn circuit ON / OFF")
    print("[W]     : Increase voltage (faster)")
    print("[S]     : Decrease voltage (slower)")
    print("[Q]     : Quit")
    
    # Circuit state variables
    circuit_on = False
    current_speed = 0.5 # Starting at your preferred sweet spot
    
    # Resolution of our test screen (later to be projector resolution)
    width, height = 800, 600
    
    while True:
        # 1. Create black canvas every frame (clear screen)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 2. Get time for smooth animations
        current_time = time.time()
        
        # 3. DRAW EFFECTS
        # Simulating a wire from a "battery" (100, 300) to an "LED" (600, 300)
        animate_wire(canvas, 100, 300, 600, 300, current_time, current_speed, circuit_on)
        draw_led(canvas, 600, 300, circuit_on)
        draw_switch(canvas, 350, 300, circuit_on)
        # Llamar a tu nueva función (Colócala donde quieras en la pantalla, ej: X=200, Y=400)
        if circuit_on:
            # Amplificamos un poco la current_speed para que afecte visualmente a la altura de la onda
            draw_oscilloscope(canvas, 200, 400, current_time, voltage_amp=40, current_amp=20 * current_speed)
        else:
            # Si está apagado, las ondas son planas (amplitud 0)
            draw_oscilloscope(canvas, 200, 400, current_time, voltage_amp=0, current_amp=0)
        
        # Add UI text to see internal data
        cv2.putText(canvas, f"State: {'ON' if circuit_on else 'OFF'}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Speed (Voltage): {current_speed:.1f}x", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 4. Show the masterpiece
        cv2.imshow("VFX Projection Test", canvas)
        
        # 5. Keyboard controls
        key = cv2.waitKey(16) & 0xFF # ~60 FPS
        
        if key == ord('q'):
            break
        elif key == ord(' '): # Spacebar
            circuit_on = not circuit_on
        elif key == ord('w'):
            current_speed += 0.1 # Finer increments
        elif key == ord('s'):
            current_speed = max(0.0, current_speed - 0.1) # Prevents negative speed

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()