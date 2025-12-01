import { useRef, useState, useEffect, useCallback } from "react";
import "./RegionSelector.css";

interface Selection {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

interface RegionSelectorProps {
  imageSrc: string;
  onRegionSelected: (regionImageData: string) => void;
}

export function RegionSelector({
  imageSrc,
  onRegionSelected,
}: RegionSelectorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const img = imageRef.current;

    if (!canvas || !ctx || !img) return;

    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;

    // Draw image
    ctx.drawImage(img, 0, 0);

    // Draw selection rectangle if exists
    if (selection) {
      const x = Math.min(selection.startX, selection.endX);
      const y = Math.min(selection.startY, selection.endY);
      const width = Math.abs(selection.endX - selection.startX);
      const height = Math.abs(selection.endY - selection.startY);

      ctx.strokeStyle = "#00ff00";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Semi-transparent overlay
      ctx.fillStyle = "rgba(0, 255, 0, 0.1)";
      ctx.fillRect(x, y, width, height);
    }
  }, [selection]);

  // Load image onto canvas
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      setImageLoaded(true);
    };
    img.src = imageSrc;
  }, [imageSrc]);

  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [imageLoaded, drawCanvas]);

  const getMousePosition = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePosition(e);
    setIsSelecting(true);
    setSelection({
      startX: pos.x,
      startY: pos.y,
      endX: pos.x,
      endY: pos.y,
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isSelecting || !selection) return;

    const pos = getMousePosition(e);
    setSelection({
      ...selection,
      endX: pos.x,
      endY: pos.y,
    });
  };

  const handleMouseUp = () => {
    setIsSelecting(false);
  };

  const handleExtractRegion = () => {
    if (!selection || !canvasRef.current || !imageRef.current) return;

    const x = Math.min(selection.startX, selection.endX);
    const y = Math.min(selection.startY, selection.endY);
    const width = Math.abs(selection.endX - selection.startX);
    const height = Math.abs(selection.endY - selection.startY);

    if (width < 5 || height < 5) {
      alert("Please select a larger region");
      return;
    }

    // Create a temporary canvas to extract the region
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext("2d");

    if (!tempCtx) return;

    // Draw only the selected region
    tempCtx.drawImage(
      imageRef.current,
      x,
      y,
      width,
      height,
      0,
      0,
      width,
      height
    );

    // Convert to base64
    const regionData = tempCanvas.toDataURL("image/png");
    onRegionSelected(regionData);
  };

  return (
    <div className="region-selector">
      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="selection-canvas"
        />
      </div>
      <div className="instructions">
        <p>Click and drag to select a region containing a neume</p>
        {selection && (
          <button onClick={handleExtractRegion} className="analyze-button">
            Analyze Selected Region
          </button>
        )}
      </div>
    </div>
  );
}
