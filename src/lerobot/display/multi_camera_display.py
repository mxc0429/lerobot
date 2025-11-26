"""Multi-camera display module for real-time visualization during teleoperation."""

import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np
from numpy.typing import NDArray


class MultiCameraDisplay:
    """
    Multi-camera display manager for real-time visualization.
    
    Runs in a separate thread to avoid blocking the control loop.
    Provides overlay rendering for camera names, FPS, timestamps, and status indicators.
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_names: List[str],
        camera_width: int = 320,
        camera_height: int = 240,
        target_fps: int = 10,
        window_name: str = "Multi-Camera View",
        show_overlays: bool = True,
    ):
        """
        Initialize the multi-camera display.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data (shared with control thread)
            camera_names: List of camera names to display
            camera_width: Width of each camera view in pixels
            camera_height: Height of each camera view in pixels
            target_fps: Target frame rate for display updates
            window_name: Name of the display window
            show_overlays: Whether to show overlays (labels, FPS, etc.)
        """
        self.model = model
        self.data = data
        self.camera_names = camera_names
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.target_fps = target_fps
        self.window_name = window_name
        self.show_overlays = show_overlays
        
        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._window_closed = False
        self._restart_on_crash = True
        self._max_consecutive_errors = 5
        self._consecutive_errors = 0
        
        # Keyboard events queue (thread-safe)
        self._keyboard_events: queue.Queue = queue.Queue()
        
        # FPS tracking
        self._frame_times: List[float] = []
        self._current_fps = 0.0
        
        # Status tracking
        self._status: Dict[str, any] = {
            "paused": False,
            "connected": True,
            "error": None,
        }
        
        # Create renderer for each camera
        self._renderers: Dict[str, mujoco.Renderer] = {}
        for cam_name in camera_names:
            # Check if camera exists in model
            try:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id >= 0:
                    renderer = mujoco.Renderer(model, camera_width, camera_height)
                    self._renderers[cam_name] = renderer
            except Exception:
                # Camera not found, will show placeholder
                pass
    
    def start(self) -> None:
        """Start the display thread."""
        if self._running:
            return
        
        self._running = True
        self._window_closed = False
        self._consecutive_errors = 0
        self._thread = threading.Thread(target=self._render_loop_with_restart, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the display thread and close the window."""
        self._running = False
        self._restart_on_crash = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # OpenCV GUI not available, ignore
            pass
    
    def is_running(self) -> bool:
        """Check if the display thread is running."""
        return self._running and not self._window_closed
    
    def is_window_closed(self) -> bool:
        """Check if the display window was closed by the operator."""
        return self._window_closed
    
    def get_keyboard_events(self) -> Dict[str, bool]:
        """
        Get keyboard events from the display window.
        
        Returns:
            Dictionary of keyboard events (e.g., {'q': True, 'p': False})
        """
        events = {}
        while not self._keyboard_events.empty():
            try:
                key, pressed = self._keyboard_events.get_nowait()
                events[key] = pressed
            except queue.Empty:
                break
        return events
    
    def set_status(self, status_key: str, value: any) -> None:
        """
        Update status indicator.
        
        Args:
            status_key: Status key (e.g., 'paused', 'connected', 'error')
            value: Status value
        """
        with self._lock:
            self._status[status_key] = value
    
    def get_fps(self) -> float:
        """Get current display FPS."""
        return self._current_fps
    
    def _render_loop_with_restart(self) -> None:
        """
        Wrapper for render loop that handles crashes and restarts.
        
        This method implements the crash recovery logic required by Requirement 7.4.
        If the display thread crashes, it will automatically restart unless:
        - The system is shutting down (_running is False)
        - Restart is disabled (_restart_on_crash is False)
        - Too many consecutive errors have occurred
        """
        while self._running and self._restart_on_crash:
            try:
                self._render_loop()
                # If render loop exits normally, break
                break
            except Exception as e:
                self._consecutive_errors += 1
                print(f"Display thread crashed: {e}")
                with self._lock:
                    self._status["error"] = f"Thread crash: {e}"
                
                # Check if we should give up after too many errors
                if self._consecutive_errors >= self._max_consecutive_errors:
                    print(f"Display thread crashed {self._consecutive_errors} times. Giving up.")
                    self._running = False
                    break
                
                # Wait a bit before restarting
                if self._running and self._restart_on_crash:
                    print(f"Restarting display thread (attempt {self._consecutive_errors})...")
                    time.sleep(0.5)
    
    def _render_loop(self) -> None:
        """Main rendering loop (runs in separate thread)."""
        frame_duration = 1.0 / self.target_fps
        
        while self._running:
            loop_start = time.perf_counter()
            
            try:
                # Render all cameras and compose grid
                composed_image = self._render_cameras()
                
                # Add overlays if enabled
                if self.show_overlays:
                    composed_image = self._add_overlays(composed_image)
                
                # Display the image
                cv2.imshow(self.window_name, composed_image)
                
                # Check if window was closed by user
                # cv2.getWindowProperty returns -1 if window was closed
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Display window closed by operator")
                    self._window_closed = True
                    self._running = False
                    # Signal window closure via keyboard events
                    self._keyboard_events.put(("window_closed", True))
                    break
                
                # Handle keyboard input (1ms wait)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # 255 means no key pressed
                    self._handle_keyboard(key)
                
                # Update FPS tracking
                self._update_fps(loop_start)
                
                # Reset error counter on successful frame
                self._consecutive_errors = 0
                
            except Exception as e:
                # Handle rendering errors gracefully
                self._consecutive_errors += 1
                print(f"Display error: {e}")
                with self._lock:
                    self._status["error"] = str(e)
                
                # If too many consecutive errors, exit the loop
                if self._consecutive_errors >= self._max_consecutive_errors:
                    print(f"Too many consecutive rendering errors ({self._consecutive_errors}). Disabling display.")
                    self._running = False
                    break
            
            # Rate limiting
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, frame_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _render_cameras(self) -> NDArray[np.uint8]:
        """
        Render all camera views and compose into grid.
        
        Returns:
            Composed image with all camera views in grid layout
        """
        camera_images = []
        
        for cam_name in self.camera_names:
            if cam_name in self._renderers:
                try:
                    # Render camera view
                    renderer = self._renderers[cam_name]
                    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    renderer.update_scene(self.data, camera=cam_id)
                    pixels = renderer.render()
                    
                    # Convert RGB to BGR for OpenCV
                    image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    camera_images.append(image)
                except Exception as e:
                    # Show placeholder on error
                    placeholder = self._create_placeholder(cam_name, f"Error: {e}")
                    camera_images.append(placeholder)
            else:
                # Camera not found, show placeholder
                placeholder = self._create_placeholder(cam_name, "Camera not found")
                camera_images.append(placeholder)
        
        # Compose into horizontal grid layout
        if len(camera_images) > 0:
            composed = np.hstack(camera_images)
        else:
            # No cameras, create empty placeholder
            composed = self._create_placeholder("No Cameras", "No cameras available")
        
        return composed
    
    def _create_placeholder(self, title: str, message: str) -> NDArray[np.uint8]:
        """
        Create a placeholder image for missing/error cameras.
        
        Args:
            title: Title text
            message: Message text
            
        Returns:
            Placeholder image
        """
        placeholder = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        placeholder[:, :] = (40, 40, 40)  # Dark gray background
        
        # Add title
        cv2.putText(
            placeholder,
            title,
            (10, self.camera_height // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        
        # Add message
        cv2.putText(
            placeholder,
            message,
            (10, self.camera_height // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        
        return placeholder
    
    def _add_overlays(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Add text overlays (FPS, labels, status, timestamp).
        
        Args:
            image: Input image
            
        Returns:
            Image with overlays added
        """
        overlay_image = image.copy()
        
        # Get current status
        with self._lock:
            status = self._status.copy()
        
        # Add camera name labels to each view
        for i, cam_name in enumerate(self.camera_names):
            x_offset = i * self.camera_width
            # Camera name label (top-left of each view)
            cv2.putText(
                overlay_image,
                cam_name,
                (x_offset + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),  # Green
                2,
                cv2.LINE_AA,
            )
        
        # Add FPS counter (top-right)
        fps_text = f"FPS: {self._current_fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        fps_x = overlay_image.shape[1] - text_size[0] - 10
        cv2.putText(
            overlay_image,
            fps_text,
            (fps_x, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),  # Yellow
            2,
            cv2.LINE_AA,
        )
        
        # Add timestamp (bottom-left)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(
            overlay_image,
            timestamp,
            (10, overlay_image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White
            1,
            cv2.LINE_AA,
        )
        
        # Add status indicators (bottom-right)
        status_y = overlay_image.shape[0] - 10
        status_x = overlay_image.shape[1] - 10
        
        # Connection status
        conn_text = "CONNECTED" if status.get("connected", False) else "DISCONNECTED"
        conn_color = (0, 255, 0) if status.get("connected", False) else (0, 0, 255)
        text_size = cv2.getTextSize(conn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        status_x -= text_size[0]
        cv2.putText(
            overlay_image,
            conn_text,
            (status_x, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            conn_color,
            1,
            cv2.LINE_AA,
        )
        
        # Paused indicator (center, large)
        if status.get("paused", False):
            paused_text = "PAUSED"
            text_size = cv2.getTextSize(paused_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            text_x = (overlay_image.shape[1] - text_size[0]) // 2
            text_y = (overlay_image.shape[0] + text_size[1]) // 2
            
            # Add semi-transparent background
            overlay = overlay_image.copy()
            cv2.rectangle(
                overlay,
                (text_x - 20, text_y - text_size[1] - 20),
                (text_x + text_size[0] + 20, text_y + 20),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.7, overlay_image, 0.3, 0, overlay_image)
            
            # Add text
            cv2.putText(
                overlay_image,
                paused_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 255),  # Yellow
                3,
                cv2.LINE_AA,
            )
        
        # Error indicator (if present)
        if status.get("error"):
            error_text = f"ERROR: {status['error']}"
            cv2.putText(
                overlay_image,
                error_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),  # Red
                1,
                cv2.LINE_AA,
            )
        
        return overlay_image
    
    def _handle_keyboard(self, key: int) -> None:
        """
        Handle keyboard input from OpenCV window.
        
        Maps keyboard keys to commands:
        - 'q': quit
        - 'p': pause/resume
        - 'r': reset
        - 's': snapshot
        - 'h': help
        
        Args:
            key: Key code from cv2.waitKey()
        """
        # Convert key code to character
        key_char = chr(key) if 0 <= key < 128 else None
        
        if key_char:
            # Map keys to command names
            key_command_map = {
                'q': 'quit',
                'p': 'pause',
                'r': 'reset',
                's': 'snapshot',
                'h': 'help',
            }
            
            # If it's a mapped command key, add the command to the queue
            if key_char in key_command_map:
                command = key_command_map[key_char]
                self._keyboard_events.put((command, True))
            else:
                # For unmapped keys, still add the raw character
                self._keyboard_events.put((key_char, True))
    
    def _update_fps(self, frame_start: float) -> None:
        """
        Update FPS tracking.
        
        Args:
            frame_start: Start time of current frame
        """
        current_time = time.perf_counter()
        self._frame_times.append(current_time)
        
        # Keep only last second of frame times
        cutoff_time = current_time - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff_time]
        
        # Calculate FPS
        if len(self._frame_times) > 1:
            time_span = self._frame_times[-1] - self._frame_times[0]
            if time_span > 0:
                self._current_fps = (len(self._frame_times) - 1) / time_span
