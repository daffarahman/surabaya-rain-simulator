export class CanvasRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private stream: MediaStream | null = null;
  private canvas: HTMLCanvasElement;
  private fps: number;

  constructor(canvas: HTMLCanvasElement, fps: number = 30) {
    this.canvas = canvas;
    this.fps = fps;
  }

  public start() {
    this.recordedChunks = [];
    
    // Capture the canvas stream
    const canvasEl = this.canvas as HTMLCanvasElement & { captureStream?: (fps?: number) => MediaStream };
    if (!canvasEl.captureStream) {
      throw new Error("canvas.captureStream is not supported on this browser.");
    }
    
    this.stream = canvasEl.captureStream(this.fps);
    
    // Supported formats check sequentially
    const types = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4;codecs=h264',
      'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
      'video/mp4',
      'video/quicktime'
    ];

    let selectedType = '';
    for (const type of types) {
      try {
        if (MediaRecorder.isTypeSupported(type)) {
          selectedType = type;
          break;
        }
      } catch {
        // isTypeSupported might throw error on some devices
      }
    }

    const options = selectedType ? { mimeType: selectedType } : undefined;
    console.log(`CanvasRecorder: Starting recording using mimeType: "${selectedType || 'default'}"`);
    try {
      this.mediaRecorder = new MediaRecorder(this.stream!, options);
    } catch (e) {
      console.warn("CanvasRecorder: Failed with preferred options, falling back to default constructor", e);
      this.mediaRecorder = new MediaRecorder(this.stream!);
    }

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        this.recordedChunks.push(event.data);
      }
    };

    this.mediaRecorder.start(100); // chunk every 100ms
  }

  public stop(): Promise<{ blob: Blob; url: string; extension: string }> {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error("MediaRecorder is not initialized."));
        return;
      }

      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder?.mimeType || 'video/webm';
        const blob = new Blob(this.recordedChunks, { type: mimeType });
        const url = URL.createObjectURL(blob);
        
        let extension = 'webm';
        if (mimeType.includes('mp4')) {
          extension = 'mp4';
        } else if (mimeType.includes('quicktime')) {
          extension = 'mov';
        }

        console.log(`CanvasRecorder: Recording stopped. Blob size: ${blob.size} bytes. MIME: ${mimeType}`);
        
        // Clean up tracks
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
        }

        resolve({ blob, url, extension });
      };

      this.mediaRecorder.stop();
    });
  }
}
