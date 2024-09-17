# Import necessary libraries
import carla
import random
import time
import numpy as np
import cv2
import os
import pickle

def main():
    """
    Main function to set up the Carla simulation environment,
    spawn a vehicle with four attached cameras, collect images from these cameras,
    process the images, and display a stitched 360-degree surround view.
    """
    # Connect to the Carla server
    client = carla.Client('localhost', 2000)  # Default IP and port
    client.set_timeout(10.0)  # Set a timeout for the client

    # Get the world object
    world = client.get_world()

    try:
        # Set up the blueprint library
        blueprint_library = world.get_blueprint_library()

        # Spawn the ego vehicle
        vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]  # Choose Tesla Model 3 as an example
        spawn_points = world.get_map().get_spawn_points()
        vehicle_spawn_point = random.choice(spawn_points)  # Randomly select a spawn point

        vehicle = world.spawn_actor(vehicle_bp, vehicle_spawn_point)
        print('Vehicle spawned')

        # Create camera sensors
        camera_blueprint = blueprint_library.find('sensor.camera.rgb')

        # Set camera attributes
        camera_blueprint.set_attribute('image_size_x', '800')  # Width of the image
        camera_blueprint.set_attribute('image_size_y', '600')  # Height of the image
        camera_blueprint.set_attribute('fov', '90')  # Field of view

        image_width = int(camera_blueprint.get_attribute("image_size_x").as_string())
        image_height = int(camera_blueprint.get_attribute("image_size_y").as_string())
        fov = float(camera_blueprint.get_attribute("fov").as_string())

        # Set up the transforms for each camera relative to the vehicle
        # Front camera
        cam_transform_front = carla.Transform(carla.Location(x=2.0, z=1.5))
        # Rear camera
        cam_transform_rear = carla.Transform(carla.Location(x=-2.0, z=1.5), carla.Rotation(yaw=180))
        # Left camera
        cam_transform_left = carla.Transform(carla.Location(y=-1.0, z=1.5), carla.Rotation(yaw=-90))
        # Right camera
        cam_transform_right = carla.Transform(carla.Location(y=1.0, z=1.5), carla.Rotation(yaw=90))

        # Spawn cameras and attach them to the vehicle
        camera_front = world.spawn_actor(camera_blueprint, cam_transform_front, attach_to=vehicle)
        camera_rear = world.spawn_actor(camera_blueprint, cam_transform_rear, attach_to=vehicle)
        camera_left = world.spawn_actor(camera_blueprint, cam_transform_left, attach_to=vehicle)
        camera_right = world.spawn_actor(camera_blueprint, cam_transform_right, attach_to=vehicle)
        print('Cameras spawned and attached to vehicle')

        # Initialize variables to hold camera images
        front_image = None
        rear_image = None
        left_image = None
        right_image = None

        # Define callback functions for each camera to process incoming images
        def front_callback(image):
            nonlocal front_image
            front_image = image_converter(image)

        def rear_callback(image):
            nonlocal rear_image
            rear_image = image_converter(image)

        def left_callback(image):
            nonlocal left_image
            left_image = image_converter(image)

        def right_callback(image):
            nonlocal right_image
            right_image = image_converter(image)

        # Set up the sensor listen methods
        camera_front.listen(front_callback)
        camera_rear.listen(rear_callback)
        camera_left.listen(left_callback)
        camera_right.listen(right_callback)

        # Start the vehicle in autopilot mode
        vehicle.set_autopilot(True)

        # Check if calibration data exists
        calibration_file = 'calibration_data.pkl'
        if os.path.exists(calibration_file):
            with open(calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)
            print('Calibration data loaded')
        else:
            # Perform calibration for each camera
            calibration_data = {}
            print('Calibration data not found, starting calibration...')
            # Capture one frame from each camera for calibration
            while front_image is None or rear_image is None or left_image is None or right_image is None:
                print('Waiting for images to calibrate...')
                world.tick()
            calibration_data['front'] = calibrate_camera(front_image, 'front')
            calibration_data['rear'] = calibrate_camera(rear_image, 'rear')
            calibration_data['left'] = calibrate_camera(left_image, 'left')
            calibration_data['right'] = calibrate_camera(right_image, 'right')
            # Save calibration data
            with open(calibration_file, 'wb') as f:
                pickle.dump(calibration_data, f)
            print('Calibration completed and data saved')

        # Run the simulation loop
        while True:
            if front_image is not None and rear_image is not None and left_image is not None and right_image is not None:
                # Process and stitch images
                stitched_image = stitch_images(front_image, rear_image, left_image, right_image, calibration_data)

                # Display the stitched image in a window
                cv2.imshow('Surround View', stitched_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit if 'q' key is pressed
            else:
                print('Waiting for all camera images...')
            world.tick()  # Advance the simulation by one tick

        # Clean up: stop and destroy sensors and the vehicle
        camera_front.stop()
        camera_rear.stop()
        camera_left.stop()
        camera_right.stop()

        camera_front.destroy()
        camera_rear.destroy()
        camera_left.destroy()
        camera_right.destroy()

        vehicle.destroy()
        print('Simulation ended and actors destroyed')

    finally:
        cv2.destroyAllWindows()  # Close all OpenCV windows

def image_converter(image):
    """
    Converts a CARLA image to a numpy array suitable for OpenCV.

    Args:
        image: The CARLA image object.

    Returns:
        A numpy array containing the image data.
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Remove alpha channel
    array = array[:, :, ::-1]  # Convert from BGRA to RGB
    return array

def calibrate_camera(image, camera_name):
    """
    Calibrates the camera by adjusting warp parameters using keyboard inputs in real-time.

    Args:
        image: The image from the camera to calibrate.
        camera_name: Name of the camera ('front', 'rear', 'left', 'right').

    Returns:
        A dictionary containing the source and destination points.
    """
    print(f'Starting calibration for {camera_name} camera.')

    # Initialize source points as the corners of the image
    h, w = image.shape[:2]
    src_points = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    # Initialize destination points as a rectangle
    offset = 100  # Some offset for initial destination points
    dst_points = np.float32([
        [offset, offset],
        [w - offset - 1, offset],
        [w - offset - 1, h - offset - 1],
        [offset, h - offset - 1]
    ])

    # Index of the selected point (0 to 3)
    selected_point = 0

    # Clone the image to display
    original_image = image.copy()

    print('Use number keys 1-4 to select a point to adjust.')
    print('Use "w", "a", "s", "d" keys to move the selected point up, left, down, right respectively.')
    print('Press "q" when done with calibration for this camera.')

    while True:
        # Compute the homography matrix
        H, _ = cv2.findHomography(src_points, dst_points)
        # Warp the image
        warped_image = cv2.warpPerspective(original_image, H, (w, h))

        # Display the warped image with points
        display_image = warped_image.copy()
        for i, (x, y) in enumerate(dst_points):
            color = (0, 0, 255) if i == selected_point else (0, 255, 0)
            cv2.circle(display_image, (int(x), int(y)), 5, color, -1)
            cv2.putText(display_image, f'{i+1}', (int(x) + 10, int(y) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow(f'Calibrate {camera_name} Camera', display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            selected_point = 0
        elif key == ord('2'):
            selected_point = 1
        elif key == ord('3'):
            selected_point = 2
        elif key == ord('4'):
            selected_point = 3
        elif key == ord('a'):
            dst_points[selected_point][0] -= 1
        elif key == ord('d'):
            dst_points[selected_point][0] += 1
        elif key == ord('w'):
            dst_points[selected_point][1] -= 1
        elif key == ord('s'):
            dst_points[selected_point][1] += 1
        elif key == ord('q'):
            print('Calibration completed for this camera.')
            break
        elif key == 27:  # Escape key
            print('Calibration cancelled.')
            break

    cv2.destroyWindow(f'Calibrate {camera_name} Camera')

    calibration = {
        'src_points': src_points,
        'dst_points': dst_points
    }
    return calibration

def stitch_images(front, rear, left, right, calibration_data):
    """
    Processes and stitches images from four cameras to create a 360-degree surround view.

    Args:
        front: Image from the front camera.
        rear: Image from the rear camera.
        left: Image from the left camera.
        right: Image from the right camera.
        calibration_data: Dictionary containing calibration data for each camera.

    Returns:
        A single image representing the stitched surround view.
    """
    # Implement the warping for each image
    bev_front = warp_image(front, calibration_data['front'])
    bev_rear = warp_image(rear, calibration_data['rear'])
    bev_left = warp_image(left, calibration_data['left'])
    bev_right = warp_image(right, calibration_data['right'])

    # Create an empty canvas to hold the stitched image
    canvas_height = 1000
    canvas_width = 1000
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Compute offsets to position each image on the canvas
    center_x = canvas_width // 2
    center_y = canvas_height // 2

    # Overlay each warped image onto the canvas
    canvas = overlay_image(canvas, bev_front, (center_x - bev_front.shape[1] // 2, center_y - bev_front.shape[0] - 50))
    canvas = overlay_image(canvas, bev_rear, (center_x - bev_rear.shape[1] // 2, center_y + 50))
    canvas = overlay_image(canvas, bev_left, (center_x - bev_left.shape[1] - 50, center_y - bev_left.shape[0] // 2))
    canvas = overlay_image(canvas, bev_right, (center_x + 50, center_y - bev_right.shape[0] // 2))

    return canvas

def warp_image(image, calibration):
    """
    Warps an image using the homography from calibration data.

    Args:
        image: The input image to warp.
        calibration: Dictionary containing 'src_points' and 'dst_points'.

    Returns:
        The warped image.
    """
    src_points = calibration['src_points']
    dst_points = calibration['dst_points']
    H, _ = cv2.findHomography(src_points, dst_points)
    h, w = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (w, h))
    return warped_image

def overlay_image(background, overlay, position):
    """
    Overlays one image onto another at a given position.

    Args:
        background: The background image.
        overlay: The image to overlay on the background.
        position: A tuple (x, y) indicating the position to place the overlay.

    Returns:
        The combined image.
    """
    x, y = position
    h, w, _ = overlay.shape

    # Check boundaries
    if y + h > background.shape[0] or x + w > background.shape[1]:
        print('Overlay image exceeds background boundaries.')
        return background

    # Overlay the image
    background[y:y+h, x:x+w] = overlay
    return background

if __name__ == '__main__':
    main()
