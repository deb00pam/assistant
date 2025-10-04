"""
Multi-Modal Input System for Truvo - Comprehensive Input Handling

This module handles various input types:
- Voice commands and speech recognition
- Screen pointing and mouse/touch events
- Gesture recognition and patterns
- Natural language goal parsing
- Keyboard shortcuts and combinations
- Combined multi-modal interactions
"""

import os
import time
import logging
import threading
import queue
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

# Import dependencies with fallbacks
try:
    import speech_recognition as sr
except ImportError:
    print("Warning: speech_recognition not installed. Install with: pip install SpeechRecognition")
    sr = None

try:
    import pynput
    from pynput import mouse, keyboard
except ImportError:
    print("Warning: pynput not installed. Install with: pip install pynput")
    pynput = None
    mouse = None
    keyboard = None

try:
    import pyaudio
except ImportError:
    print("Warning: pyaudio not installed. Install with: pip install pyaudio")
    pyaudio = None

class InputType(Enum):
    """Types of input supported by the multi-modal system"""
    VOICE = "voice"
    POINTING = "pointing" 
    GESTURE = "gesture"
    NATURAL_LANGUAGE = "natural_language"
    KEYBOARD = "keyboard"
    COMBINED = "combined"

class InputState(Enum):
    """Current state of input system"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    WAITING_FOR_CONFIRMATION = "waiting_confirmation"
    EXECUTING = "executing"

@dataclass
class InputEvent:
    """Represents a multi-modal input event"""
    input_type: InputType
    timestamp: float
    data: Any
    confidence: float
    context: Dict[str, Any]
    source_device: str = "unknown"

@dataclass
class VoiceCommand:
    """Voice command with transcribed text and metadata"""
    text: str
    confidence: float
    language: str
    timestamp: float
    audio_duration: float
    wake_word_detected: bool = False

@dataclass
class PointingEvent:
    """Screen pointing event (mouse, touch, etc.)"""
    x: int
    y: int
    action: str  # click, move, drag, scroll, etc.
    button: str  # left, right, middle, none
    timestamp: float
    pressure: float = 1.0  # For touch devices
    modifiers: List[str] = None  # ctrl, shift, alt, etc.

@dataclass
class GestureEvent:
    """Gesture recognition event"""
    gesture_type: str  # swipe, circle, pinch, zoom, etc.
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    duration: float
    confidence: float
    velocity: float
    direction: str

@dataclass
class NaturalLanguageGoal:
    """Natural language goal with parsed intent"""
    original_text: str
    parsed_intent: str
    entities: Dict[str, Any]
    action_sequence: List[Dict[str, Any]]
    confidence: float
    ambiguity_score: float

class VoiceInputHandler:
    """Handles voice input and speech recognition"""
    
    def __init__(self, wake_words: List[str] = None):
        self.wake_words = wake_words or ["hey truvo", "truvo", "computer"]
        self.recognizer = sr.Recognizer() if sr else None
        self.microphone = sr.Microphone() if sr else None
        self.is_listening = False
        self.voice_queue = queue.Queue()
        self.wake_word_sensitivity = 0.7
        
        if self.recognizer and self.microphone:
            self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                logging.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logging.info("Microphone calibrated")
        except Exception as e:
            logging.error(f"Failed to calibrate microphone: {e}")
    
    def start_listening(self):
        """Start continuous voice listening"""
        if not self.recognizer:
            logging.error("Speech recognition not available")
            return
        
        self.is_listening = True
        threading.Thread(target=self._listen_continuously, daemon=True).start()
        logging.info("Voice input started")
    
    def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        logging.info("Voice input stopped")
    
    def _listen_continuously(self):
        """Continuous listening loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Process audio in background thread
                threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()
                
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error in voice listening: {e}")
                time.sleep(0.1)
    
    def _process_audio(self, audio):
        """Process audio and extract voice commands"""
        try:
            # Transcribe audio
            text = self.recognizer.recognize_google(audio)
            confidence = 0.8  # Google doesn't provide confidence, estimate
            
            # Check for wake words
            wake_word_detected = any(wake_word in text.lower() for wake_word in self.wake_words)
            
            command = VoiceCommand(
                text=text,
                confidence=confidence,
                language="en-US",
                timestamp=time.time(),
                audio_duration=len(audio.frame_data) / audio.sample_rate,
                wake_word_detected=wake_word_detected
            )
            
            self.voice_queue.put(command)
            logging.info(f"Voice command recognized: '{text}' (wake word: {wake_word_detected})")
            
        except sr.UnknownValueError:
            # Speech was unintelligible
            pass
        except sr.RequestError as e:
            logging.error(f"Speech recognition service error: {e}")
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
    
    def get_voice_command(self, timeout: float = None) -> Optional[VoiceCommand]:
        """Get the next voice command from queue"""
        try:
            return self.voice_queue.get(timeout=timeout) if timeout else self.voice_queue.get_nowait()
        except queue.Empty:
            return None

class PointingInputHandler:
    """Handles pointing input (mouse, touch, etc.)"""
    
    def __init__(self):
        self.is_monitoring = False
        self.pointing_queue = queue.Queue()
        self.mouse_listener = None
        self.gesture_buffer = deque(maxlen=50)  # Store last 50 mouse events for gesture recognition
        self.click_threshold = 0.3  # Max time between press and release for click
        
    def start_monitoring(self):
        """Start monitoring pointing input"""
        if not mouse:
            logging.error("Mouse monitoring not available")
            return
        
        self.is_monitoring = True
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll
        )
        self.mouse_listener.start()
        logging.info("Pointing input monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring pointing input"""
        self.is_monitoring = False
        if self.mouse_listener:
            self.mouse_listener.stop()
        logging.info("Pointing input monitoring stopped")
    
    def _on_mouse_move(self, x, y):
        """Handle mouse move events"""
        if self.is_monitoring:
            event = PointingEvent(
                x=x, y=y,
                action="move",
                button="none",
                timestamp=time.time()
            )
            self.gesture_buffer.append(event)
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        if self.is_monitoring:
            action = "press" if pressed else "release"
            button_name = button.name if hasattr(button, 'name') else str(button)
            
            event = PointingEvent(
                x=x, y=y,
                action=action,
                button=button_name,
                timestamp=time.time()
            )
            
            self.pointing_queue.put(event)
            self.gesture_buffer.append(event)
            
            logging.debug(f"Mouse {action}: {button_name} at ({x}, {y})")
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events"""
        if self.is_monitoring:
            event = PointingEvent(
                x=x, y=y,
                action="scroll",
                button="wheel",
                timestamp=time.time()
            )
            # Store scroll direction in context
            event.context = {"dx": dx, "dy": dy}
            
            self.pointing_queue.put(event)
            logging.debug(f"Mouse scroll: ({dx}, {dy}) at ({x}, {y})")
    
    def get_pointing_event(self, timeout: float = None) -> Optional[PointingEvent]:
        """Get the next pointing event from queue"""
        try:
            return self.pointing_queue.get(timeout=timeout) if timeout else self.pointing_queue.get_nowait()
        except queue.Empty:
            return None

class GestureRecognizer:
    """Recognizes gestures from pointing input patterns"""
    
    def __init__(self):
        self.min_gesture_length = 5  # Minimum points for gesture
        self.max_gesture_time = 2.0  # Max time for gesture in seconds
        self.velocity_threshold = 100  # Min velocity for gesture recognition
    
    def recognize_gesture(self, point_sequence: List[PointingEvent]) -> Optional[GestureEvent]:
        """Recognize gesture from sequence of pointing events"""
        if len(point_sequence) < self.min_gesture_length:
            return None
        
        start_event = point_sequence[0]
        end_event = point_sequence[-1]
        duration = end_event.timestamp - start_event.timestamp
        
        if duration > self.max_gesture_time:
            return None
        
        # Calculate gesture properties
        distance = ((end_event.x - start_event.x) ** 2 + (end_event.y - start_event.y) ** 2) ** 0.5
        velocity = distance / duration if duration > 0 else 0
        
        if velocity < self.velocity_threshold:
            return None
        
        # Determine gesture type and direction
        gesture_type, direction = self._classify_gesture(point_sequence)
        
        if gesture_type:
            return GestureEvent(
                gesture_type=gesture_type,
                start_pos=(start_event.x, start_event.y),
                end_pos=(end_event.x, end_event.y),
                duration=duration,
                confidence=min(velocity / 500, 1.0),  # Confidence based on velocity
                velocity=velocity,
                direction=direction
            )
        
        return None
    
    def _classify_gesture(self, points: List[PointingEvent]) -> Tuple[str, str]:
        """Classify the type and direction of gesture"""
        if len(points) < 2:
            return None, None
        
        start = points[0]
        end = points[-1]
        
        dx = end.x - start.x
        dy = end.y - start.y
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        
        # Simple gesture classification
        distance = (dx**2 + dy**2)**0.5
        
        if distance > 100:
            if self._is_linear(points):
                return "swipe", direction
            elif self._is_circular(points):
                return "circle", "clockwise" if self._is_clockwise(points) else "counterclockwise"
        
        return "move", direction
    
    def _is_linear(self, points: List[PointingEvent]) -> bool:
        """Check if points form a roughly linear pattern"""
        if len(points) < 3:
            return True
        
        # Simple linearity check using variance from line
        start = points[0]
        end = points[-1]
        
        if start.x == end.x and start.y == end.y:
            return False
        
        total_deviation = 0
        for point in points[1:-1]:
            # Calculate distance from point to line
            deviation = abs((end.y - start.y) * point.x - (end.x - start.x) * point.y + 
                          end.x * start.y - end.y * start.x) / ((end.y - start.y)**2 + (end.x - start.x)**2)**0.5
            total_deviation += deviation
        
        avg_deviation = total_deviation / max(len(points) - 2, 1)
        return avg_deviation < 20  # Threshold for linearity
    
    def _is_circular(self, points: List[PointingEvent]) -> bool:
        """Check if points form a roughly circular pattern"""
        if len(points) < 8:  # Need enough points for circle
            return False
        
        # Find center point
        center_x = sum(p.x for p in points) / len(points)
        center_y = sum(p.y for p in points) / len(points)
        
        # Check if distances from center are roughly consistent
        distances = [(((p.x - center_x)**2 + (p.y - center_y)**2)**0.5) for p in points]
        avg_distance = sum(distances) / len(distances)
        
        # Check variance in distances
        variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
        return variance < (avg_distance * 0.3)**2  # 30% variance threshold
    
    def _is_clockwise(self, points: List[PointingEvent]) -> bool:
        """Check if circular gesture is clockwise"""
        if len(points) < 3:
            return True
        
        # Calculate signed area using shoelace formula
        signed_area = 0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            signed_area += (points[j].x - points[i].x) * (points[j].y + points[i].y)
        
        return signed_area > 0  # Positive area = clockwise

class NaturalLanguageProcessor:
    """Processes natural language goals and converts them to action sequences"""
    
    def __init__(self):
        self.action_keywords = {
            'click': ['click', 'press', 'tap', 'select'],
            'type': ['type', 'write', 'enter', 'input'],
            'scroll': ['scroll', 'page down', 'page up'],
            'open': ['open', 'launch', 'start', 'run'],
            'close': ['close', 'exit', 'quit', 'shut'],
            'find': ['find', 'search', 'look for', 'locate'],
            'wait': ['wait', 'pause', 'delay']
        }
        
        self.target_keywords = {
            'button': ['button', 'btn'],
            'textbox': ['textbox', 'text field', 'input field', 'text box'],
            'menu': ['menu', 'dropdown'],
            'window': ['window', 'dialog'],
            'icon': ['icon', 'shortcut'],
            'link': ['link', 'hyperlink']
        }
    
    def parse_goal(self, text: str) -> NaturalLanguageGoal:
        """Parse natural language goal into structured format"""
        text_lower = text.lower()
        
        # Extract intent and entities
        intent = self._extract_intent(text_lower)
        entities = self._extract_entities(text_lower)
        action_sequence = self._generate_action_sequence(intent, entities, text_lower)
        
        # Calculate confidence and ambiguity
        confidence = self._calculate_confidence(intent, entities, action_sequence)
        ambiguity = self._calculate_ambiguity(text_lower, entities)
        
        return NaturalLanguageGoal(
            original_text=text,
            parsed_intent=intent,
            entities=entities,
            action_sequence=action_sequence,
            confidence=confidence,
            ambiguity_score=ambiguity
        )
    
    def _extract_intent(self, text: str) -> str:
        """Extract primary intent from text"""
        for intent, keywords in self.action_keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        return "unknown"
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities (targets, values, etc.) from text"""
        entities = {}
        
        # Extract targets
        for target_type, keywords in self.target_keywords.items():
            if any(keyword in text for keyword in keywords):
                entities['target_type'] = target_type
                break
        
        # Extract quoted strings (likely text to type)
        import re
        quoted_strings = re.findall(r'"([^"]*)"', text)
        if quoted_strings:
            entities['text_content'] = quoted_strings[0]
        
        # Extract application names
        app_patterns = [
            r'open\s+(\w+)',
            r'launch\s+(\w+)',
            r'start\s+(\w+)'
        ]
        
        for pattern in app_patterns:
            match = re.search(pattern, text)
            if match:
                entities['application'] = match.group(1)
                break
        
        return entities
    
    def _generate_action_sequence(self, intent: str, entities: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
        """Generate sequence of actions from intent and entities"""
        actions = []
        
        if intent == "open" and "application" in entities:
            actions.append({
                "action_type": "key_press",
                "key": "win+r",
                "description": f"Open Run dialog"
            })
            actions.append({
                "action_type": "type",
                "text": entities["application"],
                "description": f"Type application name: {entities['application']}"
            })
            actions.append({
                "action_type": "key_press",
                "key": "enter",
                "description": "Press Enter to launch"
            })
        
        elif intent == "type" and "text_content" in entities:
            actions.append({
                "action_type": "type",
                "text": entities["text_content"],
                "description": f"Type: {entities['text_content']}"
            })
        
        elif intent == "click" and "target_type" in entities:
            actions.append({
                "action_type": "find_and_click",
                "target": entities["target_type"],
                "description": f"Find and click {entities['target_type']}"
            })
        
        # If no specific actions generated, create a generic action
        if not actions:
            actions.append({
                "action_type": "analyze_and_act",
                "goal": text,
                "description": f"Analyze screen and act on: {text}"
            })
        
        return actions
    
    def _calculate_confidence(self, intent: str, entities: Dict[str, Any], actions: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the parsing"""
        confidence = 0.5  # Base confidence
        
        if intent != "unknown":
            confidence += 0.3
        
        if entities:
            confidence += 0.2 * min(len(entities) / 3, 1.0)
        
        if actions and actions[0]["action_type"] != "analyze_and_act":
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_ambiguity(self, text: str, entities: Dict[str, Any]) -> float:
        """Calculate ambiguity score (higher = more ambiguous)"""
        ambiguity = 0.0
        
        # Multiple possible intents
        intent_matches = sum(1 for keywords in self.action_keywords.values() 
                           if any(keyword in text for keyword in keywords))
        if intent_matches > 1:
            ambiguity += 0.3
        
        # Vague language
        vague_words = ['something', 'anything', 'some', 'maybe', 'perhaps']
        if any(word in text for word in vague_words):
            ambiguity += 0.2
        
        # Missing specific targets
        if not entities.get('target_type') and not entities.get('application'):
            ambiguity += 0.3
        
        return min(ambiguity, 1.0)

class MultiModalInputManager:
    """Main manager for coordinating all input types"""
    
    def __init__(self):
        self.voice_handler = VoiceInputHandler()
        self.pointing_handler = PointingInputHandler()
        self.gesture_recognizer = GestureRecognizer()
        self.nlp_processor = NaturalLanguageProcessor()
        
        self.input_queue = queue.PriorityQueue()  # Priority queue for input events
        self.state = InputState.IDLE
        self.active_inputs = set()
        self.input_callbacks = {}
        self.context = {}
        
        # Input fusion settings
        self.voice_pointing_timeout = 3.0  # Seconds to wait for combined input
        self.gesture_buffer_size = 50
        
    def start_all_inputs(self):
        """Start monitoring all input types"""
        self.voice_handler.start_listening()
        self.pointing_handler.start_monitoring()
        self.state = InputState.LISTENING
        
        # Start input processing thread
        threading.Thread(target=self._process_inputs, daemon=True).start()
        
        logging.info("Multi-modal input system started")
    
    def stop_all_inputs(self):
        """Stop monitoring all input types"""
        self.voice_handler.stop_listening()
        self.pointing_handler.stop_monitoring()
        self.state = InputState.IDLE
        
        logging.info("Multi-modal input system stopped")
    
    def register_callback(self, input_type: InputType, callback: Callable):
        """Register callback for specific input type"""
        self.input_callbacks[input_type] = callback
    
    def _process_inputs(self):
        """Main input processing loop"""
        while self.state != InputState.IDLE:
            try:
                # Check for voice commands
                voice_cmd = self.voice_handler.get_voice_command(timeout=0.1)
                if voice_cmd:
                    self._handle_voice_input(voice_cmd)
                
                # Check for pointing events
                pointing_event = self.pointing_handler.get_pointing_event(timeout=0.1)
                if pointing_event:
                    self._handle_pointing_input(pointing_event)
                
                # Process gesture recognition
                self._process_gestures()
                
                # Process combined inputs
                self._process_combined_inputs()
                
            except Exception as e:
                logging.error(f"Error in input processing: {e}")
                time.sleep(0.1)
    
    def _handle_voice_input(self, voice_cmd: VoiceCommand):
        """Handle voice command input"""
        # Parse natural language if it's a command
        if voice_cmd.wake_word_detected or self.state == InputState.LISTENING:
            nlp_goal = self.nlp_processor.parse_goal(voice_cmd.text)
            
            input_event = InputEvent(
                input_type=InputType.VOICE,
                timestamp=voice_cmd.timestamp,
                data=nlp_goal,
                confidence=voice_cmd.confidence * nlp_goal.confidence,
                context={"wake_word": voice_cmd.wake_word_detected}
            )
            
            self._queue_input_event(input_event)
    
    def _handle_pointing_input(self, pointing_event: PointingEvent):
        """Handle pointing input event"""
        input_event = InputEvent(
            input_type=InputType.POINTING,
            timestamp=pointing_event.timestamp,
            data=pointing_event,
            confidence=1.0,  # Pointing is always certain
            context={"button": pointing_event.button, "action": pointing_event.action}
        )
        
        self._queue_input_event(input_event)
    
    def _process_gestures(self):
        """Process gesture recognition from pointing buffer"""
        if len(self.pointing_handler.gesture_buffer) >= self.gesture_recognizer.min_gesture_length:
            # Try to recognize gesture from recent points
            recent_points = list(self.pointing_handler.gesture_buffer)[-20:]  # Last 20 points
            
            gesture = self.gesture_recognizer.recognize_gesture(recent_points)
            if gesture:
                input_event = InputEvent(
                    input_type=InputType.GESTURE,
                    timestamp=gesture.start_pos[0],  # Use start time
                    data=gesture,
                    confidence=gesture.confidence,
                    context={"gesture_type": gesture.gesture_type, "direction": gesture.direction}
                )
                
                self._queue_input_event(input_event)
    
    def _process_combined_inputs(self):
        """Process combinations of different input types"""
        # This is where we'd implement fusion logic
        # For example: voice command + pointing = "click here" + point location
        pass
    
    def _queue_input_event(self, event: InputEvent):
        """Queue input event with priority"""
        # Priority based on confidence and input type
        priority = 100 - int(event.confidence * 100)  # Higher confidence = lower priority number
        
        # Adjust priority based on input type
        if event.input_type == InputType.VOICE:
            priority -= 10  # Voice has higher priority
        elif event.input_type == InputType.GESTURE:
            priority += 10  # Gestures have lower priority
        
        self.input_queue.put((priority, time.time(), event))
        
        # Call registered callback if available
        if event.input_type in self.input_callbacks:
            try:
                self.input_callbacks[event.input_type](event)
            except Exception as e:
                logging.error(f"Error in input callback: {e}")
    
    def get_next_input(self, timeout: float = None) -> Optional[InputEvent]:
        """Get the next prioritized input event"""
        try:
            if timeout:
                _, _, event = self.input_queue.get(timeout=timeout)
            else:
                _, _, event = self.input_queue.get_nowait()
            return event
        except queue.Empty:
            return None
    
    def set_context(self, key: str, value: Any):
        """Set context information for input processing"""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information"""
        return self.context.get(key, default)

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def voice_callback(event):
        print(f"Voice input: {event.data.original_text}")
    
    def pointing_callback(event):
        print(f"Pointing input: {event.data.action} at ({event.data.x}, {event.data.y})")
    
    def gesture_callback(event):
        print(f"Gesture: {event.data.gesture_type} {event.data.direction}")
    
    # Create and start multi-modal input system
    input_manager = MultiModalInputManager()
    
    # Register callbacks
    input_manager.register_callback(InputType.VOICE, voice_callback)
    input_manager.register_callback(InputType.POINTING, pointing_callback)
    input_manager.register_callback(InputType.GESTURE, gesture_callback)
    
    # Start monitoring
    input_manager.start_all_inputs()
    
    try:
        print("Multi-modal input system running. Try voice commands, mouse gestures, etc.")
        print("Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping multi-modal input system...")
        input_manager.stop_all_inputs()