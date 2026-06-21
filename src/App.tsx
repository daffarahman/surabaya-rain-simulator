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

interface FloatingItem {
  imageIndex: number;
  xPercent: number;
  size: number;
  phase: number;
  rotationOffset: number;
}

function wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let currentLine = words[0] || '';

  for (let i = 1; i < words.length; i++) {
    const word = words[i];
    const width = ctx.measureText(currentLine + " " + word).width;
    if (width < maxWidth) {
      currentLine += " " + word;
    } else {
      lines.push(currentLine);
      currentLine = word;
    }
  }
  lines.push(currentLine);
  return lines;
}

const tipsData = [
  {
    image: '/img/tips/reusable_tumbler_bag.png',
    text: 'Gunakan tumbler dan tote bag reusable',
  },
  {
    image: '/img/tips/minimal_packaging.png',
    text: 'Pilih produk dengan kemasan minim sampah',
  },
  {
    image: '/img/tips/waste_sorting.png',
    text: 'Pilah sampah sebelum dibuang',
  },
  {
    image: '/img/tips/dont_litter_sewer.png',
    text: 'Jangan buang sampah ke selokan',
  },
  {
    image: '/img/tips/conscious_shopping.png',
    text: 'Kurangi belanja impulsif (overconsumption): Beli seperlunya, bukan karena tren sesaat',
  },
  {
    image: '/img/tips/reuse_items.png',
    text: 'Pakai kembali barang yang masih layak',
  },
  {
    image: '/img/tips/clean_environment.png',
    text: 'Ikut menjaga kebersihan lingkungan sekitar',
  },
];

export default function App() {
  // --- State Management ---
  const [loadingText, setLoadingText] = useState('Initializing WASM Resolvers...');
  const [isLoading, setIsLoading] = useState(true);
  const facingMode = 'user';
  const [isRecording, setIsRecording] = useState(false);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [replayModalOpen, setReplayModalOpen] = useState(false);
  const [consoleLogs, setConsoleLogs] = useState<ConsoleLogMessage[]>([]);
  const currentMode = 'simulator';
  const [simState, setSimState] = useState<number>(0);
  const [activeTipIndex, setActiveTipIndex] = useState<number>(0);

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const carouselRef = useRef<HTMLDivElement | null>(null);

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

  // Refs for animation loop performance (to avoid tearing down renderLoop on state change)
  const currentModeRef = useRef<'shower' | 'simulator' | null>(null);
  const simStateRef = useRef<number>(0);
  const floodHeightRef = useRef<number>(0);

  // Preloaded trash images
  const trashImagesRef = useRef<HTMLImageElement[]>([]);
  // Preloaded tip images for canvas recorder
  const tipImagesRef = useRef<HTMLImageElement[]>([]);

  // Floating items specifications (increased quantity and sizes)
  const floatingItemsRef = useRef<FloatingItem[]>([
    { imageIndex: 0, xPercent: 0.10, size: 55, phase: 0, rotationOffset: 0.1 },
    { imageIndex: 1, xPercent: 0.22, size: 50, phase: Math.PI * 0.3, rotationOffset: -0.05 },
    { imageIndex: 2, xPercent: 0.32, size: 62, phase: Math.PI * 0.65, rotationOffset: 0.08 },
    { imageIndex: 3, xPercent: 0.44, size: 48, phase: Math.PI * 1.1, rotationOffset: -0.12 },
    { imageIndex: 4, xPercent: 0.54, size: 58, phase: Math.PI * 1.45, rotationOffset: 0.03 },
    { imageIndex: 0, xPercent: 0.64, size: 52, phase: Math.PI * 0.15, rotationOffset: -0.06 },
    { imageIndex: 1, xPercent: 0.74, size: 65, phase: Math.PI * 0.85, rotationOffset: 0.05 },
    { imageIndex: 2, xPercent: 0.84, size: 46, phase: Math.PI * 1.6, rotationOffset: -0.1 },
    { imageIndex: 3, xPercent: 0.92, size: 54, phase: Math.PI * 0.4, rotationOffset: 0.08 },
    { imageIndex: 4, xPercent: 0.05, size: 49, phase: Math.PI * 1.2, rotationOffset: -0.05 },
  ]);

  // Preload images on mount
  useEffect(() => {
    const loadedImages: HTMLImageElement[] = [];
    const srcList = [
      '/img/trashes/1.png',
      '/img/trashes/2.png',
      '/img/trashes/3.png',
      '/img/trashes/4.png',
      '/img/trashes/5.png',
    ];

    srcList.forEach((src, idx) => {
      const img = new Image();
      img.src = src;
      img.onload = () => {
        loadedImages[idx] = img;
        console.log(`Trash image ${src} loaded successfully`);
      };
    });
    trashImagesRef.current = loadedImages;

    const loadedTipImages: HTMLImageElement[] = [];
    tipsData.forEach((tip, idx) => {
      const img = new Image();
      img.src = tip.image;
      img.onload = () => {
        loadedTipImages[idx] = img;
        console.log(`Tip image ${tip.image} loaded successfully`);
      };
    });
    tipImagesRef.current = loadedTipImages;
  }, []);

  useEffect(() => {
    currentModeRef.current = currentMode;
  }, [currentMode]);

  useEffect(() => {
    simStateRef.current = simState;
  }, [simState]);

  useEffect(() => {
    if (simState !== 8) {
      setActiveTipIndex(0);
    }
  }, [simState]);

  const activeTipIndexRef = useRef<number>(0);
  useEffect(() => {
    activeTipIndexRef.current = activeTipIndex;
  }, [activeTipIndex]);

  const isRecordingRef = useRef<boolean>(false);
  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

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

    // Handle loadedmetadata and window resize
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      console.log(`Canvas resized to window dimensions: ${canvas.width}x${canvas.height}`);
    };
    window.addEventListener('resize', handleResize);
    video.addEventListener('loadedmetadata', handleResize);
    // Initial size setting
    handleResize();

    const renderLoop = () => {
      if (video.readyState >= 2) {
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        const mirrored = facingMode === 'user';
        const timestamp = video.currentTime * 1000;

        // Calculate aspect cover cropping parameters
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        const videoRatio = videoWidth / videoHeight;
        const canvasRatio = canvasWidth / canvasHeight;

        let sx = 0;
        let sy = 0;
        let sWidth = videoWidth;
        let sHeight = videoHeight;

        if (videoRatio > canvasRatio) {
          sWidth = videoHeight * canvasRatio;
          sx = (videoWidth - sWidth) / 2;
        } else {
          sHeight = videoWidth / canvasRatio;
          sy = (videoHeight - sHeight) / 2;
        }

        // Render camera frame matching the cover crop
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        if (mirrored) {
          ctx.save();
          ctx.translate(canvasWidth, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, canvasWidth, canvasHeight);
          ctx.restore();
        } else {
          ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, canvasWidth, canvasHeight);
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
            const tips = [4, 8, 12, 16, 20];
            for (const tipIndex of tips) {
              const landmark = landmarks[tipIndex];
              if (landmark) {
                // Map landmark normalized video coordinates to covered canvas coordinates
                const normalizedX = mirrored ? 1 - landmark.x : landmark.x;
                const x = ((normalizedX * videoWidth) - sx) * (canvasWidth / sWidth);
                const y = ((landmark.y * videoHeight) - sy) * (canvasHeight / sHeight);

                ctx.beginPath();
                ctx.arc(x, y, 22, 0, Math.PI * 2);
                ctx.stroke();

                if (tipIndex === 8) {
                  const handleX = canvasWidth - 95;
                  const dx = x - handleX;
                  const dy = y - pullString.handleY;
                  const dist = Math.sqrt(dx * dx + dy * dy);

                  if (dist < 75) {
                    if (dist < minGrabDist) {
                      minGrabDist = dist;
                      closestHandFinger = { x, y };
                    }
                  } else if (pullString.isGrabbing) {
                    if (Math.abs(dx) < 180 && dy > -100 && dy < 300) {
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
        const currentModeVal = currentModeRef.current;
        const simStateVal = simStateRef.current;

        if (closestHandFinger) {
          pullString.isGrabbing = true;
          pullString.handleY = Math.max(
            pullString.restY,
            Math.min(closestHandFinger.y, pullString.restY + 160)
          );

          if (pullString.handleY - pullString.restY > 90) {
            if (!pullString.hasToggledThisPull) {
              pullString.hasToggledThisPull = true;
              if (currentModeVal === 'shower') {
                pullString.stateOn = !pullString.stateOn;
                console.log(`Switch pulled. Rain state: ${pullString.stateOn ? 'ON' : 'OFF'}`);
              } else if (currentModeVal === 'simulator') {
                if (simStateVal === 8) {
                  handleReset();
                } else {
                  setSimState((prev) => {
                    const next = (prev + 1) % 9;
                    console.log(`Simulator Stage Advanced: ${prev} -> ${next}`);
                    return next;
                  });
                }
              }
            }
          } else if (pullString.handleY - pullString.restY < 55) {
            pullString.hasToggledThisPull = false;
          }
        } else {
          pullString.isGrabbing = false;
          pullString.hasToggledThisPull = false;

          const force = -pullString.k * (pullString.handleY - pullString.restY);
          pullString.vy += force;
          pullString.vy *= pullString.damping;
          pullString.handleY += pullString.vy;
        }

        // Determine rain parameters based on current mode and state
        let isRaining = false;
        let spawnRateBase = 0;
        let spawnChance = 0;
        let pSizeMin = 5.0;
        let pSizeRange = 5.0;

        if (currentModeVal === 'shower') {
          isRaining = pullString.stateOn;
          spawnRateBase = 3;
          spawnChance = 0.125; // average 3.125
        } else if (currentModeVal === 'simulator') {
          if (simStateVal === 1 || simStateVal === 2) {
            isRaining = true;
            spawnRateBase = 1;
            spawnChance = 0.5; // average 1.5
            pSizeMin = 3.0;
            pSizeRange = 3.0;
          } else if (simStateVal === 3 || simStateVal === 4) {
            isRaining = true;
            spawnRateBase = 3;
            spawnChance = 0.0; // average 3.0
            pSizeMin = 5.0;
            pSizeRange = 4.0;
          } else if (simStateVal === 5 || simStateVal === 6 || simStateVal === 7) {
            isRaining = true;
            spawnRateBase = 6;
            spawnChance = 0.5; // average 6.5
            pSizeMin = 7.5;
            pSizeRange = 5.5;
          }
        }

        // Spawn Water Particles
        if (isRaining) {
          const spawnCount = spawnRateBase + (Math.random() < spawnChance ? 1 : 0);
          for (let i = 0; i < spawnCount; i++) {
            particlesRef.current.push({
              x: Math.random() * canvasWidth,
              y: 0,
              vx: (Math.random() - 0.5) * 1.2,
              vy: Math.random() * 2 + 5,
              size: Math.random() * pSizeRange + pSizeMin,
              opacity: Math.random() * 0.4 + 0.5,
            });
          }
        }

        // Update flood height
        if (currentModeVal === 'simulator' && (simStateVal === 5 || simStateVal === 6 || simStateVal === 7)) {
          const maxFlood = canvasHeight * 0.75;
          if (floodHeightRef.current < maxFlood) {
            floodHeightRef.current = Math.min(maxFlood, floodHeightRef.current + 1.2);
          }

          // Auto-transitions based on screen height percentage
          if (simStateVal === 5 && floodHeightRef.current >= canvasHeight * 0.45) {
            simStateRef.current = 6;
            setSimState(6);
            console.log('Auto-advanced to Stage 6 (water reached 45% screen height)');
          } else if (simStateVal === 6 && floodHeightRef.current >= canvasHeight * 0.55) {
            simStateRef.current = 7;
            setSimState(7);
            console.log('Auto-advanced to Stage 7 (water reached 55% screen height)');
          }
        } else {
          if (floodHeightRef.current > 0) {
            floodHeightRef.current -= 3.0;
            if (floodHeightRef.current < 0) floodHeightRef.current = 0;
          }
        }

        // Update & Render Water Particles
        const currentParticles = particlesRef.current;
        const nextParticles: WaterParticle[] = [];
        const currentFloodY = canvasHeight - floodHeightRef.current;
        const waterTimePhase = timestamp ? (timestamp * 0.003) : 0;

        ctx.save();
        // Setup glowing blue effect for particles
        ctx.shadowColor = 'rgba(0, 191, 255, 0.8)';
        ctx.shadowBlur = 8;
        for (const p of currentParticles) {
          p.x += p.vx;
          p.y += p.vy;
          p.vy += 0.22; // gravity force

          let alive = true;

          // Check if hitting the flood level (wavy surface)
          const wave1 = Math.sin(p.x * 0.015 + waterTimePhase) * 3.5;
          const wave2 = Math.cos(p.x * 0.035 - waterTimePhase * 0.7) * 1.5;
          const exactFloodY = currentFloodY + wave1 + wave2;

          if (p.y >= exactFloodY) {
            alive = false;
            // Spawn splash on flood surface
            const splashCount = Math.floor(Math.random() * 2) + 2;
            for (let s = 0; s < splashCount; s++) {
              splashParticlesRef.current.push({
                x: p.x,
                y: exactFloodY - 2,
                vx: (Math.random() - 0.5) * 3.0,
                vy: -Math.random() * 1.5 - 0.8,
                size: p.size * 0.5,
                opacity: p.opacity,
                life: Math.floor(Math.random() * 6) + 6,
              });
            }
          } else if (p.y >= 0 && p.x >= 0 && p.x < canvasWidth) {
            // Check selfie segmentation mask collision under cover-cropping scale
            const maskWidth = segmentationWidthRef.current;
            const maskHeight = segmentationHeightRef.current;

            if (mask && maskWidth > 0 && maskHeight > 0) {
              // Map canvas coordinates to normalized video coordinates
              const normX = (sx + (p.x / canvasWidth) * sWidth) / videoWidth;
              const normY = (sy + (p.y / canvasHeight) * sHeight) / videoHeight;

              if (normX >= 0 && normX <= 1 && normY >= 0 && normY <= 1) {
                const mx = Math.floor(normX * maskWidth);
                const my = Math.floor(normY * maskHeight);
                const rawMx = mirrored ? maskWidth - 1 - mx : mx;
                const maskIndex = my * maskWidth + rawMx;

                if (maskIndex >= 0 && maskIndex < mask.length && mask[maskIndex] > 0.5) {
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
                    // 55% Chance: Slide along silhouette boundary
                    let nearestEdgeX = -1;
                    for (let dx = 1; dx <= 6; dx++) {
                      const leftMx = mx - dx;
                      const rightMx = mx + dx;
                      let leftValid = false;
                      let rightValid = false;

                      if (leftMx >= 0) {
                        const rLMx = mirrored ? maskWidth - 1 - leftMx : leftMx;
                        const idxL = my * maskWidth + rLMx;
                        if (mask[idxL] <= 0.5) leftValid = true;
                      }

                      if (rightMx < maskWidth) {
                        const rRMx = mirrored ? maskWidth - 1 - rightMx : rightMx;
                        const idxR = my * maskWidth + rRMx;
                        if (mask[idxR] <= 0.5) rightValid = true;
                      }

                      if (leftValid && rightValid) {
                        nearestEdgeX = p.vx < 0 ? leftMx : rightMx;
                        break;
                      } else if (leftValid) {
                        nearestEdgeX = leftMx;
                        break;
                      } else if (rightValid) {
                        nearestEdgeX = rightMx;
                        break;
                      }
                    }

                    if (nearestEdgeX !== -1) {
                      // Map back to canvas coordinates
                      const canvasNearestEdgeX = ((nearestEdgeX / maskWidth) * videoWidth - sx) * (canvasWidth / sWidth);
                      p.x = canvasNearestEdgeX;
                      p.vx = (canvasNearestEdgeX - p.x) * 0.12 + (Math.random() - 0.5) * 0.4;
                      p.vy = p.vy * 0.35 + 1.2; // Slide down slowly
                    } else {
                      // Fall inside with slower velocity
                      p.vy = 1.8;
                      p.vx = (Math.random() - 0.5) * 1.5;
                    }
                  }
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
            ctx.moveTo(p.x - p.vx * 1.8, p.y - p.vy * 1.8);
            ctx.lineTo(p.x, p.y);
            ctx.strokeStyle = `rgba(0, 191, 255, ${p.opacity * 0.95})`;
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

          const wave1 = Math.sin(sp.x * 0.015 + waterTimePhase) * 3.5;
          const wave2 = Math.cos(sp.x * 0.035 - waterTimePhase * 0.7) * 1.5;
          const exactFloodY = currentFloodY + wave1 + wave2;

          // Splash particle should die if it falls below the wavy flood surface
          if (sp.life > 0 && sp.y < exactFloodY && sp.y < canvasHeight && sp.x >= 0 && sp.x < canvasWidth) {
            nextSplashes.push(sp);

            // Draw splash dots
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, sp.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 191, 255, ${(sp.life / 16) * 0.8})`;
            ctx.fill();
          }
        }
        splashParticlesRef.current = nextSplashes;
        ctx.restore();

        // Draw Wavy Flood Water
        if (floodHeightRef.current > 0) {
          ctx.save();

          // 1. Draw solid wavy water body
          ctx.beginPath();
          ctx.moveTo(0, canvasHeight);
          for (let x = 0; x <= canvasWidth; x += 5) {
            const wave1 = Math.sin(x * 0.015 + waterTimePhase) * 3.5;
            const wave2 = Math.cos(x * 0.035 - waterTimePhase * 0.7) * 1.5;
            const currY = currentFloodY + wave1 + wave2;
            ctx.lineTo(x, currY);
          }
          ctx.lineTo(canvasWidth, canvasHeight);
          ctx.closePath();
          ctx.fillStyle = 'rgba(30, 144, 255, 0.35)';
          ctx.fill();

          // 3. Draw floating items bobbing and riding the waves
          for (const item of floatingItemsRef.current) {
            const itemX = item.xPercent * canvasWidth;
            const wave1 = Math.sin(itemX * 0.015 + waterTimePhase) * 3.5;
            const wave2 = Math.cos(itemX * 0.035 - waterTimePhase * 0.7) * 1.5;
            const exactFloodY = currentFloodY + wave1 + wave2;

            // Bobbing movement (minor vertical bounce)
            const bob = Math.sin(waterTimePhase * 1.2 + item.phase) * 3.0;
            const floatY = exactFloodY + bob - item.size * 0.3; // align slightly above wave line

            const img = trashImagesRef.current[item.imageIndex];
            if (img && img.complete) {
              ctx.save();
              ctx.translate(itemX, floatY);
              // Small rotational sway
              const angle = Math.sin(waterTimePhase * 0.8 + item.phase) * 0.08 + item.rotationOffset;
              ctx.rotate(angle);
              ctx.drawImage(img, -item.size / 2, -item.size / 2, item.size, item.size);
              ctx.restore();
            }
          }

          ctx.restore();
        }

        // Draw Pull String Switch (only if mode is selected)
        if (currentModeVal !== null) {
          ctx.save();
          const switchX = canvasWidth - 95;
          const rectW = 120;
          const rectH = 42;
          const rectX = switchX - rectW / 2;
          const rectY = pullString.handleY - rectH / 2;
          const radius = 12;

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
          ctx.font = "bold 14px 'Montserrat', sans-serif";
          ctx.fillStyle = pullString.isGrabbing ? '#1a1a24' : '#ffffff';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          let knobText = '';
          if (currentModeVal === 'shower') {
            knobText = pullString.stateOn ? 'ON' : 'OFF';
          } else {
            if (simStateVal === 0) knobText = 'PULL';
            else if (simStateVal === 7) knobText = 'STOP FLOOD';
            else if (simStateVal === 8) knobText = 'RESET';
            else knobText = `STAGE ${simStateVal}`;
          }

          ctx.fillText(
            knobText,
            switchX,
            pullString.handleY + 1
          );
          ctx.restore();
        }

        // Draw overlays on canvas if recording is active so they are captured in the video
        if (isRecordingRef.current) {
          const isMobile = canvasWidth < 600;

          // 1. Draw brand overlay title (bottom-left)
          ctx.save();
          ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
          ctx.shadowBlur = 8;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 2;
          ctx.fillStyle = '#ffffff';
          const brandSize = isMobile ? 11 : 14;
          ctx.font = `700 ${brandSize}px 'Montserrat', sans-serif`;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'bottom';
          ctx.fillText('🌧️ Surabaya Rain Simulator 🌧️', isMobile ? 16 : 24, canvasHeight - 24);
          ctx.restore();

          // 2. Draw rain intensity overlay (bottom-right)
          if (currentModeVal === 'simulator') {
            ctx.save();
            const percentageText = `🌧️ ${simStateVal === 0 ? '0%' :
                                      simStateVal === 1 ? '15%' :
                                      simStateVal === 2 ? '30%' :
                                      simStateVal === 3 ? '50%' :
                                      simStateVal === 4 ? '65%' :
                                      simStateVal === 5 ? '80%' :
                                      simStateVal === 6 ? '95%' :
                                      simStateVal === 7 ? '100%' : '0%'}`;
            const pctSize = isMobile ? 24 : 32;
            ctx.font = `800 ${pctSize}px 'Montserrat', sans-serif`;
            
            const padX = isMobile ? 12 : 16;
            const padY = isMobile ? 6 : 8;
            const pctMetrics = ctx.measureText(percentageText);
            const pctTextW = pctMetrics.width;
            const pctRectW = pctTextW + padX * 2;
            const pctRectH = pctSize + padY * 2;
            const pctRectX = canvasWidth - (isMobile ? 16 : 24) - pctRectW;
            const pctRectY = canvasHeight - 24 - pctRectH;

            ctx.beginPath();
            ctx.roundRect(pctRectX, pctRectY, pctRectW, pctRectH, 16);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
            ctx.fill();

            ctx.fillStyle = '#1a1a24';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(percentageText, pctRectX + pctRectW / 2, pctRectY + pctRectH / 2);
            ctx.restore();
          }

          // 3. Draw docked info box stay (top-left)
          if (currentModeVal === 'simulator' && (simStateVal === 2 || simStateVal === 4 || simStateVal === 6)) {
            ctx.save();
            const stayX = isMobile ? 16 : 24;
            const stayY = 64;
            const stayW = isMobile ? (canvasWidth - 32) : 360;
            const stayPad = isMobile ? 16 : 22;

            let bgColor = '';
            let shadowColor = '';
            let titleColor = '';
            let titleText = '';
            let subtitleText = '';
            let bodyLinesRaw: string[] = [];

            if (simStateVal === 2) {
              bgColor = 'rgba(255, 250, 215, 0.96)';
              shadowColor = 'rgba(234, 179, 8, 0.45)';
              titleColor = '#b45309';
              titleText = '☁️ Curah Hujan Ringan';
              subtitleText = 'ITS Area – Kondisi Aman';
              bodyLinesRaw = [
                'Gerimis tipis khas Surabaya Timur.',
                'Air masih tertampung normal oleh drainase ITS–Keputih.',
                'Risiko genangan sangat rendah, aktivitas tetap lancar.'
              ];
            } else if (simStateVal === 4) {
              bgColor = 'rgba(255, 240, 222, 0.96)';
              shadowColor = 'rgba(249, 115, 22, 0.45)';
              titleColor = '#c2410c';
              titleText = '🌧️ Curah Hujan Sedang';
              subtitleText = 'Waspada Genangan Lokal';
              bodyLinesRaw = [
                'Hujan mulai stabil dan lebih lama turun.',
                'Beberapa titik rendah di sekitar Keputih–Sukolilo bisa tergenang sementara.',
                'Drainase mulai bekerja lebih berat, perlu kewaspadaan.'
              ];
            } else if (simStateVal === 6) {
              bgColor = 'rgba(246, 235, 255, 0.96)';
              shadowColor = 'rgba(168, 85, 247, 0.45)';
              titleColor = '#7e22ce';
              titleText = '⛈️ Curah Hujan Deras';
              subtitleText = 'Zona Rawan Genangan ITS & Sekitar';
              bodyLinesRaw = [
                'Hujan lebat dalam durasi panjang.',
                'Area Surabaya Timur seperti Keputih, Mulyorejo, dan sekitar ITS berpotensi banjir lokal.',
                'Air bisa naik cepat karena kapasitas saluran terbatas dan aliran tersumbat di beberapa titik.'
              ];
            }

            const titleSize = 13;
            const subtitleSize = 11;
            const bodySize = 11.5;
            const titleLineH = 18;
            const subtitleLineH = 16;
            const bodyLineH = 18.4;

            ctx.font = `bold ${titleSize}px 'Montserrat', sans-serif`;
            const titleLines = wrapText(ctx, titleText, stayW - stayPad * 2);

            ctx.font = `600 ${subtitleSize}px 'Montserrat', sans-serif`;
            const subtitleLines = wrapText(ctx, subtitleText, stayW - stayPad * 2);
            
            ctx.font = `${bodySize}px 'Montserrat', sans-serif`;
            const allBodyLines: { text: string; isFirst: boolean }[] = [];
            for (const rawLine of bodyLinesRaw) {
              // Indent the wrapped text width by 14px to accommodate bullet point
              const wrapped = wrapText(ctx, rawLine, stayW - stayPad * 2 - 14);
              wrapped.forEach((line, idx) => {
                allBodyLines.push({ text: line, isFirst: idx === 0 });
              });
            }

            const titleH = titleLines.length * titleLineH;
            const subtitleH = subtitleLines.length * subtitleLineH;
            const bodyH = allBodyLines.length * bodyLineH;
            const gap = 8;
            const stayH = stayPad * 2 + titleH + gap + subtitleH + gap + bodyH;

            ctx.shadowColor = shadowColor;
            ctx.shadowBlur = 48;
            ctx.fillStyle = bgColor;
            ctx.beginPath();
            ctx.roundRect(stayX, stayY, stayW, stayH, 20);
            ctx.fill();
            ctx.shadowBlur = 0;

            ctx.fillStyle = titleColor;
            ctx.font = `bold ${titleSize}px 'Montserrat', sans-serif`;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            let currY = stayY + stayPad;
            for (const line of titleLines) {
              ctx.fillText(line, stayX + stayPad, currY);
              currY += titleLineH;
            }

            ctx.fillStyle = titleColor;
            ctx.font = `600 ${subtitleSize}px 'Montserrat', sans-serif`;
            currY += 2;
            for (const line of subtitleLines) {
              ctx.fillText(line, stayX + stayPad, currY);
              currY += subtitleLineH;
            }

            ctx.fillStyle = '#1a1a24';
            ctx.font = `${bodySize}px 'Montserrat', sans-serif`;
            currY += gap;
            for (const line of allBodyLines) {
              if (line.isFirst) {
                ctx.save();
                ctx.fillStyle = titleColor; // Match color tone
                ctx.beginPath();
                ctx.arc(stayX + stayPad + 3, currY + bodyLineH / 2 - 1, 2.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
              }
              // Draw indented text
              ctx.fillText(line.text, stayX + stayPad + 14, currY);
              currY += bodyLineH;
            }
            ctx.restore();
          }

          // 4. Draw Stage 8 popup carousel modal (center)
          if (currentModeVal === 'simulator' && simStateVal === 8) {
            ctx.save();
            ctx.fillStyle = 'rgba(26, 26, 36, 0.35)';
            ctx.fillRect(0, 0, canvasWidth, canvasHeight);

            const cardW = isMobile ? (canvasWidth - 32) : 400;
            const cardH = 300;
            const cardX = (canvasWidth - cardW) / 2;
            const cardY = (canvasHeight - cardH) / 2;
            ctx.beginPath();
            ctx.roundRect(cardX, cardY, cardW, cardH, 24);
            ctx.fillStyle = '#ffffff';
            ctx.fill();

            ctx.fillStyle = '#1a1a24';
            ctx.font = "bold 13px 'Montserrat', sans-serif";
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Cegah Banjir dari Kebiasaan Sehari-hari', canvasWidth / 2, cardY + 24);

            const img = tipImagesRef.current[activeTipIndexRef.current];
            const imgSize = 80;
            const imgX = (canvasWidth - imgSize) / 2;
            const imgY = cardY + 54;
            if (img && img.complete) {
              ctx.drawImage(img, imgX, imgY, imgSize, imgSize);
            }

            ctx.fillStyle = '#1a1a24';
            ctx.font = "600 11px 'Montserrat', sans-serif";
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            const tipText = tipsData[activeTipIndexRef.current]?.text || '';
            const tipTextLines = wrapText(ctx, tipText, cardW - 80);
            let currY = imgY + imgSize + 16;
            for (const line of tipTextLines) {
              ctx.fillText(line, canvasWidth / 2, currY);
              currY += 16;
            }

            // Draw "Paham" button if last slide
            if (activeTipIndexRef.current === tipsData.length - 1) {
              ctx.save();
              ctx.fillStyle = 'rgba(26, 26, 36, 0.5)';
              ctx.font = "500 9px 'Montserrat', sans-serif";
              ctx.fillText('Klik Paham untuk reset simulator', canvasWidth / 2, currY + 6);

              const btnW = 100;
              const btnH = 32;
              const btnX = (canvasWidth - btnW) / 2;
              const btnY = cardY + 222;

              ctx.beginPath();
              ctx.roundRect(btnX, btnY, btnW, btnH, 8);
              ctx.fillStyle = '#1a1a24';
              ctx.fill();

              ctx.fillStyle = '#ffffff';
              ctx.font = "bold 11px 'Montserrat', sans-serif";
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText('Paham', canvasWidth / 2, btnY + btnH / 2);
              ctx.restore();
            }

            ctx.font = "bold 18px 'Montserrat', sans-serif";
            ctx.fillStyle = activeTipIndexRef.current === 0 ? 'rgba(26, 26, 36, 0.25)' : '#1a1a24';
            ctx.fillText('‹', cardX + 24, cardY + cardH / 2);

            ctx.fillStyle = activeTipIndexRef.current === tipsData.length - 1 ? 'rgba(26, 26, 36, 0.25)' : '#1a1a24';
            ctx.fillText('›', cardX + cardW - 24, cardY + cardH / 2);

            const dotRadius = 3;
            const dotGap = 8;
            const dotsTotalW = (tipsData.length - 1) * dotGap;
            const startDotX = (canvasWidth - dotsTotalW) / 2;
            const dotY = cardY + cardH - 24;

            for (let i = 0; i < tipsData.length; i++) {
              ctx.beginPath();
              ctx.arc(startDotX + i * dotGap, dotY, dotRadius, 0, Math.PI * 2);
              ctx.fillStyle = activeTipIndexRef.current === i ? '#1a1a24' : '#e5e5e7';
              ctx.fill();
            }
            ctx.restore();
          }
        }


      }

      animationFrameIdRef.current = requestAnimationFrame(renderLoop);
    };

    // Run animation frame loop
    animationFrameIdRef.current = requestAnimationFrame(renderLoop);

    return () => {
      window.removeEventListener('resize', handleResize);
      video.removeEventListener('loadedmetadata', handleResize);
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [isLoading, facingMode]);

  const handleCarouselScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    const container = e.currentTarget;
    if (container.clientWidth > 0) {
      const index = Math.round(container.scrollLeft / container.clientWidth);
      setActiveTipIndex(index);
    }
  }, []);

  const scrollPrev = useCallback(() => {
    const container = carouselRef.current;
    if (container && container.clientWidth > 0) {
      const nextIndex = Math.max(0, activeTipIndex - 1);
      container.scrollTo({
        left: nextIndex * container.clientWidth,
        behavior: 'smooth',
      });
      setActiveTipIndex(nextIndex);
    }
  }, [activeTipIndex]);

  const scrollNext = useCallback(() => {
    const container = carouselRef.current;
    if (container && container.clientWidth > 0) {
      const nextIndex = Math.min(tipsData.length - 1, activeTipIndex + 1);
      container.scrollTo({
        left: nextIndex * container.clientWidth,
        behavior: 'smooth',
      });
      setActiveTipIndex(nextIndex);
    }
  }, [activeTipIndex]);

  // --- Reset controls ---
  const handleReset = useCallback(async () => {
    console.log('Resetting simulation state to 0');
    setSimState(0);
    setActiveTipIndex(0);

    if (isRecordingRef.current) {
      console.log('Stopping recording on simulator reset...');
      if (canvasRecorderRef.current) {
        try {
          isRecordingRef.current = false;
          const { blob, url } = await canvasRecorderRef.current.stop();
          setIsRecording(false);

          // Launch review modal
          setRecordedBlob(blob);
          setRecordingUrl(url);
          setReplayModalOpen(true);
          console.log('Recording stopped on reset. Preview ready.');
        } catch (err) {
          console.error('Failed to stop recording on reset:', err);
          setIsRecording(false);
        }
      }
    }
  }, []);

  // --- Recording controls ---
  const toggleRecording = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (isRecordingRef.current) {
      // Stop Recording
      console.log('Stopping recording...');
      if (canvasRecorderRef.current) {
        try {
          isRecordingRef.current = false;
          const { blob, url } = await canvasRecorderRef.current.stop();
          setIsRecording(false);

          // Launch review modal
          setRecordedBlob(blob);
          setRecordingUrl(url);
          setReplayModalOpen(true);
          console.log('Recording stopped. Preview ready.');
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
        isRecordingRef.current = true;
        recorder.start();
        setIsRecording(true);
        console.log('Recording started.');
      } catch (err) {
        console.error('Failed to start recording:', err);
      }
    }
  }, []);

  const downloadRecording = useCallback((format: 'mp4' | 'webm') => {
    if (!recordedBlob) return;

    const downloadBlob = new Blob([recordedBlob], {
      type: format === 'mp4' ? 'video/mp4' : 'video/webm'
    });
    const url = URL.createObjectURL(downloadBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `surabaya-rain-${Date.now()}.${format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setTimeout(() => URL.revokeObjectURL(url), 100);
    console.log(`Recording downloaded manually as ${format.toUpperCase()}`);
  }, [recordedBlob]);

  // --- Keyboard Event Handlers ---
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toUpperCase();

      if (key === 'S') {
        const pullString = pullStringSwitchRef.current;
        // Trigger bounce animation
        pullString.handleY = pullString.restY + 100;

        const mode = currentModeRef.current;
        if (mode === 'shower') {
          pullString.stateOn = !pullString.stateOn;
          console.log(`Keyboard toggle! Rain state: ${pullString.stateOn ? 'ON' : 'OFF'}`);
        } else if (mode === 'simulator') {
          if (simStateRef.current === 8) {
            handleReset();
          } else {
            setSimState((prev) => {
              const next = (prev + 1) % 9;
              console.log(`Keyboard Simulator Stage Advanced: ${prev} -> ${next}`);
              return next;
            });
          }
        }
      } else if (key === 'R') {
        toggleRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [toggleRecording, handleReset]);



  // --- Cleanup on unmount ---
  useEffect(() => {
    return () => {
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isRecordingRef.current) return;
    if (simState !== 8) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // Scale client coordinates to canvas internal pixel dimensions
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

    const isMobile = canvas.width < 600;
    const cardW = isMobile ? (canvas.width - 32) : 400;
    const cardH = 300;
    const cardX = (canvas.width - cardW) / 2;
    const cardY = (canvas.height - cardH) / 2;

    // Left Arrow Click Range
    const leftArrowXMin = cardX + 10;
    const leftArrowXMax = cardX + 50;
    const arrowYMin = cardY + cardH / 2 - 30;
    const arrowYMax = cardY + cardH / 2 + 30;

    // Right Arrow Click Range
    const rightArrowXMin = cardX + cardW - 50;
    const rightArrowXMax = cardX + cardW - 10;

    if (y >= arrowYMin && y <= arrowYMax) {
      if (x >= leftArrowXMin && x <= leftArrowXMax && activeTipIndex > 0) {
        const nextIdx = activeTipIndex - 1;
        setActiveTipIndex(nextIdx);
        activeTipIndexRef.current = nextIdx;
        console.log('Canvas Carousel Prev Clicked', nextIdx);
      } else if (x >= rightArrowXMin && x <= rightArrowXMax && activeTipIndex < tipsData.length - 1) {
        const nextIdx = activeTipIndex + 1;
        setActiveTipIndex(nextIdx);
        activeTipIndexRef.current = nextIdx;
        console.log('Canvas Carousel Next Clicked', nextIdx);
      }
    }

    // Paham Button Click Range (only on last tip)
    if (activeTipIndex === tipsData.length - 1) {
      const btnW = 100;
      const btnH = 32;
      const btnX = (canvas.width - btnW) / 2;
      const btnY = cardY + 222;

      if (x >= btnX && x <= btnX + btnW && y >= btnY && y <= btnY + btnH) {
        console.log('Canvas Carousel Paham Clicked');
        handleReset();
      }
    }
  };

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
      <canvas ref={canvasRef} className="ar-canvas" onClick={handleCanvasClick} />

      {/* Hidden camera preview */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="raw-video"
      />

      {/* Camera Shutter Record Button */}
      {!isLoading && (
        <button
          className={`shutter-btn ${isRecording ? 'recording' : ''}`}
          onClick={toggleRecording}
          aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
        >
          <span className="shutter-inner" />
        </button>
      )}

      {/* Docked Info Box for Surabaya Rain Simulator */}
      {!isLoading && !isRecording && currentMode === 'simulator' && (simState === 2 || simState === 4 || simState === 6) && (
        <div className={`info-box-stay theme-${simState === 2 ? 'yellow' : simState === 4 ? 'orange' : 'purple'}`}>
          {simState === 2 && (
            <>
              <h3 className="info-box-stay-title">☁️ Curah Hujan Ringan</h3>
              <p className="info-box-stay-subtitle">ITS Area – Kondisi Aman</p>
              <ul className="info-box-stay-list">
                <li>Gerimis tipis khas Surabaya Timur.</li>
                <li>Air masih tertampung normal oleh drainase ITS–Keputih.</li>
                <li>Risiko genangan sangat rendah, aktivitas tetap lancar.</li>
              </ul>
            </>
          )}

          {simState === 4 && (
            <>
              <h3 className="info-box-stay-title">🌧️ Curah Hujan Sedang</h3>
              <p className="info-box-stay-subtitle">Waspada Genangan Lokal</p>
              <ul className="info-box-stay-list">
                <li>Hujan mulai stabil dan lebih lama turun.</li>
                <li>Beberapa titik rendah di sekitar Keputih–Sukolilo bisa tergenang sementara.</li>
                <li>Drainase mulai bekerja lebih berat, perlu kewaspadaan.</li>
              </ul>
            </>
          )}

          {simState === 6 && (
            <>
              <h3 className="info-box-stay-title">⛈️ Curah Hujan Deras</h3>
              <p className="info-box-stay-subtitle">Zona Rawan Genangan ITS & Sekitar</p>
              <ul className="info-box-stay-list">
                <li>Hujan lebat dalam durasi panjang.</li>
                <li>Area Surabaya Timur seperti Keputih, Mulyorejo, dan sekitar ITS berpotensi banjir lokal.</li>
                <li>Air bisa naik cepat karena kapasitas saluran terbatas dan aliran tersumbat di beberapa titik.</li>
              </ul>
            </>
          )}
        </div>
      )}

      {/* Centered Popup Checklist Modal for Stage 8 */}
      {!isLoading && !isRecording && currentMode === 'simulator' && simState === 8 && (
        <div className="info-box-backdrop no-overlay">
          <div className="popup-layout-wrapper">
            <div className="info-box-card">
              <h3 className="info-box-title">Cegah Banjir dari Kebiasaan Sehari-hari</h3>
              <div className="tips-carousel-container">
                {/* Left Arrow Button */}
                <button
                  className="carousel-arrow left"
                  onClick={scrollPrev}
                  aria-label="Previous Tip"
                  disabled={activeTipIndex === 0}
                >
                  ‹
                </button>

                <div
                  ref={carouselRef}
                  className="tips-carousel-scroll"
                  onScroll={handleCarouselScroll}
                >
                  {tipsData.map((tip, index) => (
                    <div key={index} className="tip-carousel-slide">
                      <img src={tip.image} alt={tip.text} className="tip-slide-img" />
                      <p className="tip-slide-text">{tip.text}</p>
                      {index === tipsData.length - 1 && (
                        <>
                          <span style={{ fontSize: '9px', opacity: 0.6, marginTop: '-8px', marginBottom: '4px' }}>
                            Klik Paham untuk reset simulator
                          </span>
                          <button className="btn-paham" onClick={handleReset}>
                            Paham
                          </button>
                        </>
                      )}
                    </div>
                  ))}
                </div>

                {/* Right Arrow Button */}
                <button
                  className="carousel-arrow right"
                  onClick={scrollNext}
                  aria-label="Next Tip"
                  disabled={activeTipIndex === tipsData.length - 1}
                >
                  ›
                </button>

                <div className="carousel-dots">
                  {tipsData.map((_, index) => (
                    <div
                      key={index}
                      className={`carousel-dot ${activeTipIndex === index ? 'active' : ''}`}
                      onClick={() => {
                        const container = carouselRef.current;
                        if (container) {
                          container.scrollTo({
                            left: index * container.clientWidth,
                            behavior: 'smooth',
                          });
                        }
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Precipitation / Rain Intensity Overlay */}
      {!isLoading && !isRecording && currentMode === 'simulator' && (
        <div className="rain-intensity-overlay">
          🌧️ {simState === 0 ? '0%' :
            simState === 1 ? '15%' :
              simState === 2 ? '30%' :
                simState === 3 ? '50%' :
                  simState === 4 ? '65%' :
                    simState === 5 ? '80%' :
                      simState === 6 ? '95%' :
                        simState === 7 ? '100%' : '0%'}
        </div>
      )}

      {/* Brand Overlay Title */}
      {!isLoading && !isRecording && (
        <div className="brand-overlay-title">
          🌧️ Surabaya Rain Simulator 🌧️
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
                className="btn-download-mp4"
                onClick={() => downloadRecording('mp4')}
              >
                Save as MP4
              </button>
              <button
                className="btn-download-webm"
                onClick={() => downloadRecording('webm')}
              >
                Save as WebM
              </button>
              <button
                className="btn-minimal"
                onClick={() => {
                  setReplayModalOpen(false);
                  setRecordingUrl(null);
                  setRecordedBlob(null);
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
