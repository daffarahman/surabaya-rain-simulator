import { useState, useEffect, useRef, useCallback } from 'react';
import { FilesetResolver, HandLandmarker, ImageSegmenter, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import { CanvasRecorder } from './utils/CanvasRecorder';

interface WaterParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  isSplash?: boolean;
}

interface SplashParticle extends WaterParticle {
  life: number;
}

interface ConsoleLogMessage {
  type: 'log' | 'warn' | 'error';
  text: string;
  id: string;
}

export default function App() {
  // --- State Management ---
  const [loadingText, setLoadingText] = useState('Initializing WASM Resolvers...');
  const [isLoading, setIsLoading] = useState(true);
  const facingMode = 'user';
  const [isRecording, setIsRecording] = useState(false);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [replayModalOpen, setReplayModalOpen] = useState(false);
  const [consoleLogs, setConsoleLogs] = useState<ConsoleLogMessage[]>([]);

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const imageSegmenterRef = useRef<ImageSegmenter | null>(null);

  // Pull string switch configurations
  const pullStringSwitchRef = useRef({
    stateOn: false,
    handleY: 120,
    vy: 0,
    restY: 120,
    k: 0.15,      // stiffness
    damping: 0.82, // dampening
    isGrabbing: false,
    hasToggledThisPull: false,
  });

  const canvasRecorderRef = useRef<CanvasRecorder | null>(null);
  const particlesRef = useRef<WaterParticle[]>([]);
  const splashParticlesRef = useRef<SplashParticle[]>([]);
  const activeStreamRef = useRef<MediaStream | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);

  // Segmentation mask cache refs
  const segmentationMaskRef = useRef<Float32Array | null>(null);
  const segmentationWidthRef = useRef<number>(0);
  const segmentationHeightRef = useRef<number>(0);

  // --- Dynamic Console Logging Hook ---
  useEffect(() => {
    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;

    const addLogMessage = (type: 'log' | 'warn' | 'error', args: unknown[]) => {
      const text = args
        .map((arg) => (typeof arg === 'object' ? JSON.stringify(arg) : String(arg)))
        .join(' ');
      setConsoleLogs((prev) => [
        ...prev.slice(-30),
        { type, text, id: `${Date.now()}-${Math.random()}` },
      ]);
    };

    console.log = (...args) => {
      originalLog(...args);
      addLogMessage('log', args);
    };
    console.warn = (...args) => {
      originalWarn(...args);
      addLogMessage('warn', args);
    };
    console.error = (...args) => {
      originalError(...args);
      addLogMessage('error', args);
    };

    return () => {
      console.log = originalLog;
      console.warn = originalWarn;
      console.error = originalError;
    };
  }, []);



  // --- MediaPipe Models Loader ---
  useEffect(() => {
    let active = true;

    async function loadModels() {
      try {
        setLoadingText('Loading WASM Modules...');
        console.log('Loading MediaPipe FilesetResolver...');
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
        );

        if (!active) return;

        setLoadingText('Loading Hand Landmarker Model...');
        let handLandmarker;
        try {
          console.log('Initializing Hand Landmarker (GPU)...');
          handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
          });
        } catch (gpuError) {
          console.warn('GPU delegate failed for Hand Landmarker, falling back to CPU...', gpuError);
          if (!active) return;
          handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
          });
        }

        if (!active) return;
        handLandmarkerRef.current = handLandmarker;

        setLoadingText('Loading Image Segmenter Model...');
        let imageSegmenter;
        try {
          console.log('Initializing Image Segmenter (GPU)...');
          imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite',
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            outputCategoryMask: false,
            outputConfidenceMasks: true,
          });
        } catch (gpuError) {
          console.warn('GPU delegate failed for Image Segmenter, falling back to CPU...', gpuError);
          if (!active) return;
          imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite',
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            outputCategoryMask: false,
            outputConfidenceMasks: true,
          });
        }

        if (!active) return;
        imageSegmenterRef.current = imageSegmenter;

        setLoadingText('Opening Camera Stream...');
        setIsLoading(false);
        console.log('MediaPipe Models loaded successfully.');
      } catch (err: unknown) {
        console.error('Initialization error:', err);
        const errMsg = err instanceof Error ? err.message : String(err);
        setLoadingText(`Error: ${errMsg}`);
      }
    }

    loadModels();

    return () => {
      active = false;
    };
  }, []);

  // --- Webcam Stream Startup ---
  useEffect(() => {
    if (isLoading) return;

    let active = true;

    async function startCamera() {
      try {
        if (activeStreamRef.current) {
          activeStreamRef.current.getTracks().forEach((track) => track.stop());
        }

        console.log(`Starting camera stream facing "${facingMode}"...`);
        const constraints: MediaStreamConstraints = {
          video: {
            facingMode: facingMode,
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        activeStreamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err: unknown) {
        console.error('Camera stream access failed:', err);
      }
    }

    startCamera();

    return () => {
      active = false;
    };
  }, [isLoading, facingMode]);

  // --- Animation loop & processing ---
  useEffect(() => {
    if (isLoading) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastVideoTime = -1;

    // Handle loadedmetadata resizing
    const handleLoadedMetadata = () => {
      if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        console.log(`Canvas resized to video dimensions: ${canvas.width}x${canvas.height}`);
      }
    };
    video.addEventListener('loadedmetadata', handleLoadedMetadata);

    const renderLoop = () => {
      if (video.readyState >= 2) {
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        const mirrored = facingMode === 'user';
        const timestamp = video.currentTime * 1000;

        // Render camera frame
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        if (mirrored) {
          ctx.save();
          ctx.translate(canvasWidth, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
          ctx.restore();
        } else {
          ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
        }

        // Process MediaPipe Models
        let handLandmarksList: NormalizedLandmark[][] = [];
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;

          // Hand tracking
          if (handLandmarkerRef.current) {
            const handResult = handLandmarkerRef.current.detectForVideo(video, timestamp);
            if (handResult && handResult.landmarks) {
              handLandmarksList = handResult.landmarks;
            }
          }

          // Image segmenter
          if (imageSegmenterRef.current) {
            imageSegmenterRef.current.segmentForVideo(video, timestamp, (result) => {
              if (result && result.confidenceMasks && result.confidenceMasks.length > 0) {
                const confidenceMask = result.confidenceMasks[0];
                segmentationMaskRef.current = confidenceMask.getAsFloat32Array();
                segmentationWidthRef.current = confidenceMask.width;
                segmentationHeightRef.current = confidenceMask.height;
              }
            });
          }
        }

        const mask = segmentationMaskRef.current;

        // Update Pull String handle & Spring physics
        const pullString = pullStringSwitchRef.current;
        let closestHandFinger: { x: number; y: number } | null = null;
        let minGrabDist = Infinity;

        // Draw Minimalist Hand Tips Visualizer
        if (handLandmarksList.length > 0) {
          ctx.save();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#ffffff';

          for (const landmarks of handLandmarksList) {
            // Find tips of the 5 fingers (indices: 4, 8, 12, 16, 20)
            const tips = [4, 8, 12, 16, 20];
            for (const tipIndex of tips) {
              const landmark = landmarks[tipIndex];
              if (landmark) {
                const x = (mirrored ? 1 - landmark.x : landmark.x) * canvasWidth;
                const y = landmark.y * canvasHeight;

                // Draw circles (radius 9, stroke-only)
                ctx.beginPath();
                ctx.arc(x, y, 9, 0, Math.PI * 2);
                ctx.stroke();

                // Track closest index finger (tip index 8) to switch handle
                if (tipIndex === 8) {
                  const handleX = canvasWidth - 50;
                  const dx = x - handleX;
                  const dy = y - pullString.handleY;
                  const dist = Math.sqrt(dx * dx + dy * dy);

                  if (dist < 40) {
                    if (dist < minGrabDist) {
                      minGrabDist = dist;
                      closestHandFinger = { x, y };
                    }
                  } else if (pullString.isGrabbing) {
                    // Pulling follow threshold
                    if (Math.abs(dx) < 90 && dy > -50 && dy < 160) {
                      closestHandFinger = { x, y };
                    }
                  }
                }
              }
            }
          }
          ctx.restore();
        }

        // Grabbing switch updates
        if (closestHandFinger) {
          pullString.isGrabbing = true;
          pullString.handleY = Math.max(
            pullString.restY,
            Math.min(closestHandFinger.y, pullString.restY + 120)
          );

          if (pullString.handleY - pullString.restY > 65) {
            if (!pullString.hasToggledThisPull) {
              pullString.stateOn = !pullString.stateOn;
              pullString.hasToggledThisPull = true;
              console.log(`Switch pulled. Rain state: ${pullString.stateOn ? 'ON' : 'OFF'}`);
            }
          } else if (pullString.handleY - pullString.restY < 40) {
            // Allow re-toggling within the same drag if returned high
            pullString.hasToggledThisPull = false;
          }
        } else {
          pullString.isGrabbing = false;
          pullString.hasToggledThisPull = false;

          // Spring physics formula application
          const force = -pullString.k * (pullString.handleY - pullString.restY);
          pullString.vy += force;
          pullString.vy *= pullString.damping;
          pullString.handleY += pullString.vy;
        }

        // Spawn Water Particles
        if (pullString.stateOn) {
          // Spawn rate: increased by 25% (average 3.125 drops per frame)
          const spawnCount = Math.random() < 0.875 ? 3 : 4;
          for (let i = 0; i < spawnCount; i++) {
            particlesRef.current.push({
              x: Math.random() * canvasWidth,
              y: 0,
              vx: (Math.random() - 0.5) * 1.2,
              vy: Math.random() * 2 + 5,
              size: Math.random() * 2.5 + 2.5,
              opacity: Math.random() * 0.4 + 0.5,
            });
          }
        }

        // Update & Render Water Particles
        const currentParticles = particlesRef.current;
        const nextParticles: WaterParticle[] = [];

        ctx.save();
        for (const p of currentParticles) {
          p.x += p.vx;
          p.y += p.vy;
          p.vy += 0.22; // gravity force

          let alive = true;

          // Check if within bounds for mask check
          if (p.y >= 0 && p.y < canvasHeight && p.x >= 0 && p.x < canvasWidth) {
            const px = Math.floor(p.x);
            const py = Math.floor(p.y);
            const rawX = mirrored ? canvasWidth - 1 - px : px;
            const index = py * canvasWidth + rawX;

            if (mask && index >= 0 && index < mask.length && mask[index] > 0.5) {
              const roll = Math.random();
              if (roll < 0.20) {
                // 20% Chance: Die on impact + spawn splashes
                alive = false;
                const splashCount = Math.floor(Math.random() * 3) + 3;
                for (let s = 0; s < splashCount; s++) {
                  splashParticlesRef.current.push({
                    x: p.x,
                    y: p.y - 2,
                    vx: (Math.random() - 0.5) * 3.5,
                    vy: -Math.random() * 2 - 1.2,
                    size: p.size * 0.6,
                    opacity: p.opacity,
                    life: Math.floor(Math.random() * 8) + 8,
                  });
                }
              } else if (roll < 0.45) {
                // 25% Chance: Bounce back
                p.vy = -p.vy * 0.25;
                p.vx = (Math.random() - 0.5) * 3.5;
                p.y -= 2; // Offset upward to escape continuous collision
              } else {
                // 55% Chance: Slide along silhouette boundary (scanRange = 16)
                let nearestEdgeX = -1;
                for (let dx = 1; dx <= 16; dx++) {
                  const leftX = px - dx;
                  const rightX = px + dx;
                  let leftValid = false;
                  let rightValid = false;

                  if (leftX >= 0) {
                    const rLX = mirrored ? canvasWidth - 1 - leftX : leftX;
                    const idxL = py * canvasWidth + rLX;
                    if (mask[idxL] <= 0.5) leftValid = true;
                  }

                  if (rightX < canvasWidth) {
                    const rRX = mirrored ? canvasWidth - 1 - rightX : rightX;
                    const idxR = py * canvasWidth + rRX;
                    if (mask[idxR] <= 0.5) rightValid = true;
                  }

                  if (leftValid && rightValid) {
                    // Match flow velocity direction
                    nearestEdgeX = p.vx < 0 ? leftX : rightX;
                    break;
                  } else if (leftValid) {
                    nearestEdgeX = leftX;
                    break;
                  } else if (rightValid) {
                    nearestEdgeX = rightX;
                    break;
                  }
                }

                if (nearestEdgeX !== -1) {
                  p.x = nearestEdgeX;
                  p.vx = (nearestEdgeX - px) * 0.12 + (Math.random() - 0.5) * 0.4;
                  p.vy = p.vy * 0.35 + 1.2; // Slide down slowly
                } else {
                  // Fall inside with slower velocity
                  p.vy = 1.8;
                  p.vx = (Math.random() - 0.5) * 1.5;
                }
              }
            }
          }

          // Off-screen checks
          if (p.y > canvasHeight || p.x < 0 || p.x > canvasWidth) {
            alive = false;
          }

          if (alive) {
            nextParticles.push(p);

            // Draw rectangle rain particle (slanted streak)
            ctx.beginPath();
            ctx.moveTo(p.x - p.vx * 1.2, p.y - p.vy * 1.2);
            ctx.lineTo(p.x, p.y);
            ctx.strokeStyle = `rgba(255, 255, 255, ${p.opacity * 0.8})`;
            ctx.lineWidth = p.size;
            ctx.stroke();
          }
        }
        particlesRef.current = nextParticles;

        // Update & Render Splash Particles
        const currentSplashes = splashParticlesRef.current;
        const nextSplashes: SplashParticle[] = [];

        for (const sp of currentSplashes) {
          sp.x += sp.vx;
          sp.y += sp.vy;
          sp.vy += 0.22; // gravity
          sp.life -= 1;

          if (sp.life > 0 && sp.y < canvasHeight && sp.x >= 0 && sp.x < canvasWidth) {
            nextSplashes.push(sp);

            // Draw splash dots
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, sp.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${(sp.life / 16) * 0.7})`;
            ctx.fill();
          }
        }
        splashParticlesRef.current = nextSplashes;
        ctx.restore();

        // Draw Pull String Switch
        ctx.save();
        const switchX = canvasWidth - 50;
        const rectW = 54;
        const rectH = 22;
        const rectX = switchX - rectW / 2;
        const rectY = pullString.handleY - rectH / 2;
        const radius = 6;

        // Draw the vertical string line with wiggle effect based on velocity and time
        ctx.beginPath();
        const segments = 15;
        ctx.moveTo(switchX, 0);
        for (let i = 1; i <= segments; i++) {
          const t = i / segments;
          const currY = rectY * t;
          const envelope = Math.sin(Math.PI * t);
          const timePhase = timestamp ? (timestamp * 0.02) : 0;
          // Wiggle amplitude scales with velocity, plus a tiny idle/dragging breeze effect
          const wiggle = (pullString.vy * 0.4 + (pullString.isGrabbing ? 0.2 : 0.01)) * envelope * Math.sin(t * Math.PI * 3 + timePhase);
          ctx.lineTo(switchX + wiggle, currY);
        }
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Draw the handle knob as a rounded rectangle with white outline and transparent/white fill
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(rectX, rectY, rectW, rectH, radius);
        } else {
          ctx.moveTo(rectX + radius, rectY);
          ctx.lineTo(rectX + rectW - radius, rectY);
          ctx.quadraticCurveTo(rectX + rectW, rectY, rectX + rectW, rectY + radius);
          ctx.lineTo(rectX + rectW, rectY + rectH - radius);
          ctx.quadraticCurveTo(rectX + rectW, rectY + rectH, rectX + rectW - radius, rectY + rectH);
          ctx.lineTo(rectX + radius, rectY + rectH);
          ctx.quadraticCurveTo(rectX, rectY + rectH, rectX, rectY + rectH - radius);
          ctx.lineTo(rectX, rectY + radius);
          ctx.quadraticCurveTo(rectX, rectY, rectX + radius, rectY);
          ctx.closePath();
        }
        if (pullString.isGrabbing) {
          ctx.fillStyle = '#ffffff';
          ctx.fill();
        }
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.8;
        ctx.stroke();

        // Draw state label inside the rounded rectangle
        ctx.font = "bold 9px 'JetBrains Mono', monospace";
        ctx.fillStyle = pullString.isGrabbing ? '#1a1a24' : '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          pullString.stateOn ? 'ON' : 'OFF',
          switchX,
          pullString.handleY + 1
        );
        ctx.restore();

        // Draw Minimalist REC dot
        if (isRecording) {
          ctx.save();
          ctx.font = "bold 11px 'JetBrains Mono', monospace";
          ctx.fillStyle = '#1a1a24';
          ctx.textAlign = 'left';
          ctx.fillText('• REC', 24, 32);
          ctx.restore();
        }
      }

      animationFrameIdRef.current = requestAnimationFrame(renderLoop);
    };

    // Run animation frame loop
    animationFrameIdRef.current = requestAnimationFrame(renderLoop);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [isLoading, facingMode, isRecording]);

  // --- Recording controls ---
  const toggleRecording = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (isRecording) {
      // Stop Recording
      console.log('Stopping recording...');
      if (canvasRecorderRef.current) {
        try {
          const { url, extension } = await canvasRecorderRef.current.stop();
          setIsRecording(false);

          // Auto-download trigger
          const link = document.createElement('a');
          link.href = url;
          link.download = `surabaya-rain-${Date.now()}.${extension}`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          // Launch review modal
          setRecordingUrl(url);
          setReplayModalOpen(true);
          console.log(`Recording downloaded: surabaya-rain-${Date.now()}.${extension}`);
        } catch (err) {
          console.error('Failed to stop recording:', err);
          setIsRecording(false);
        }
      }
    } else {
      // Start Recording
      console.log('Initiating canvas recording...');
      try {
        const recorder = new CanvasRecorder(canvas, 30);
        canvasRecorderRef.current = recorder;
        recorder.start();
        setIsRecording(true);
        console.log('Recording started.');
      } catch (err) {
        console.error('Failed to start recording:', err);
      }
    }
  }, [isRecording]);

  // --- Keyboard Event Handlers ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toUpperCase();

      if (key === 'S') {
        const pullString = pullStringSwitchRef.current;
        pullString.stateOn = !pullString.stateOn;
        // Trigger bounce animation
        pullString.handleY = pullString.restY + 70;
        console.log(`Keyboard toggle! Rain state: ${pullString.stateOn ? 'ON' : 'OFF'}`);
      } else if (key === 'R') {
        toggleRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [toggleRecording]);



  // --- Cleanup on unmount ---
  useEffect(() => {
    return () => {
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="app-container">
      {/* Loading Overlay */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loader-spinner" />
          <div className="loader-text">{loadingText}</div>
          <div className="loading-logs">
            {consoleLogs.slice(-6).map((log) => (
              <div key={log.id} className="loading-log-item">
                &gt; {log.text}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Canvas */}
      <canvas ref={canvasRef} className="ar-canvas" />

      {/* Hidden camera preview */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="raw-video"
      />



      {/* UI Overlay Controls */}
      {!isLoading && (
        <div className="controls-overlay">
          <button
            className={`btn-minimal ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
          >
            {isRecording ? 'STOP REC' : 'REC'}
          </button>
        </div>
      )}

      {/* Mobile Console Overlay */}
      <div className="console-overlay">
        <div className="console-header">
          <span>DEBUG LOGS</span>
          <button
            className="console-clear-btn"
            onClick={() => setConsoleLogs([])}
          >
            CLEAR
          </button>
        </div>
        <ul className="console-logs-list">
          {consoleLogs.map((log) => (
            <li key={log.id} className={`log-${log.type}`}>
              [{new Date().toLocaleTimeString()}] {log.text}
            </li>
          ))}
          {consoleLogs.length === 0 && (
            <li className="log-info" style={{ opacity: 0.5 }}>
              No debug output yet. Pull the string or flip cameras to log events.
            </li>
          )}
        </ul>
      </div>

      {/* Replay Clip Modal */}
      {replayModalOpen && recordingUrl && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <h3 className="modal-title">Recording Preview</h3>
            <video
              src={recordingUrl}
              controls
              autoPlay
              loop
              className="modal-video"
            />
            <div className="modal-actions">
              <button
                className="btn-minimal"
                onClick={() => {
                  setReplayModalOpen(false);
                  setRecordingUrl(null);
                }}
              >
                CLOSE
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
