# LockerLink AI Integration Guide

This guide explains how to integrate the LockerLink AI microservice with your existing LockerLink application.

## Architecture

```
LockerLink (Next.js) → Cloudinary (Video Storage) → LockerLink AI Service → Firestore (Analysis Results)
```

## Setup

### 1. Deploy the AI Microservice

Deploy the `lockerlink-ai` service to your infrastructure (e.g., AWS, GCP, or local server):

```bash
# Example: Run on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or use Docker:
```bash
docker run --gpus all -p 8000:8000 lockerlink-ai:latest
```

Set the service URL in your LockerLink environment variables:
```env
NEXT_PUBLIC_AI_SERVICE_URL=http://localhost:8000
# Or for production:
NEXT_PUBLIC_AI_SERVICE_URL=https://ai-service.lockerlink.com
```

### 2. Install Integration Utilities

Copy the integration utility file to your LockerLink project:

```bash
# Copy the utility file
cp lockerlink-ai/integration/lockerlink-ai-client.ts ../lockerlink/utils/
```

### 3. Update Firestore Schema

Add analysis fields to your `highlights` collection:

```typescript
interface Highlight {
  // ... existing fields ...
  
  // AI Analysis fields
  aiAnalysis?: {
    status: "pending" | "processing" | "completed" | "failed";
    completedAt?: number;
    metrics?: {
      vertical_jump?: number;
      approach_speed?: number;
      ball_touch_detected?: boolean;
      kill_accuracy?: number;
      max_reach_height?: number;
      action_type?: "kill" | "block" | "other";
    };
    raw_data?: {
      frames_analyzed?: number;
      player_detected?: boolean;
      ball_detected?: boolean;
    };
  };
}
```

## Integration Methods

### Method 1: Automatic Analysis on Upload (Recommended)

Automatically trigger analysis when a highlight is uploaded.

**Update `app/highlights/page.tsx`:**

```typescript
import { analyzeVideo } from "@/utils/lockerlink-ai-client";

// In handleUpload function, after video upload:
const handleUpload = async () => {
  // ... existing upload code ...
  
  // After adding highlight to Firestore:
  const highlightRef = await addDoc(collection(db, "highlights"), {
    // ... existing fields ...
    aiAnalysis: {
      status: "pending"
    }
  });
  
  // Trigger analysis asynchronously
  analyzeVideo(highlightRef.id, videoUpload.secureUrl, user.uid)
    .catch(error => {
      console.error("AI analysis failed:", error);
      // Update status to failed
      updateDoc(highlightRef, {
        "aiAnalysis.status": "failed"
      });
    });
};
```

### Method 2: Manual Analysis Trigger

Add a button to manually trigger analysis for existing highlights.

**Add to `app/highlights/[id]/page.tsx`:**

```typescript
import { analyzeVideo } from "@/utils/lockerlink-ai-client";

const handleAnalyzeVideo = async () => {
  if (!highlight?.videoURL) return;
  
  try {
    // Update status to processing
    await updateDoc(doc(db, "highlights", highlight.id), {
      "aiAnalysis.status": "processing"
    });
    
    // Trigger analysis
    const result = await analyzeVideo(highlight.id, highlight.videoURL, highlight.userId);
    
    // Update Firestore with results
    await updateDoc(doc(db, "highlights", highlight.id), {
      "aiAnalysis": {
        status: "completed",
        completedAt: Date.now(),
        metrics: result.metrics,
        raw_data: result.analysis
      }
    });
    
    alert("Video analysis completed!");
  } catch (error) {
    console.error("Analysis failed:", error);
    await updateDoc(doc(db, "highlights", highlight.id), {
      "aiAnalysis.status": "failed"
    });
    alert("Analysis failed. Please try again.");
  }
};
```

### Method 3: Background Job (Cloud Functions)

For production, use Firebase Cloud Functions to process videos in the background.

**Create `functions/src/analyzeHighlight.ts`:**

```typescript
import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import axios from "axios";

const AI_SERVICE_URL = functions.config().ai_service.url;

export const onHighlightCreated = functions.firestore
  .document("highlights/{highlightId}")
  .onCreate(async (snap, context) => {
    const highlight = snap.data();
    const highlightId = context.params.highlightId;
    
    if (!highlight.videoURL) {
      return null;
    }
    
    try {
      // Update status to processing
      await snap.ref.update({
        "aiAnalysis.status": "processing"
      });
      
      // Call AI service
      const response = await axios.post(
        `${AI_SERVICE_URL}/analyze/video`,
        {
          video_url: highlight.videoURL,
          highlight_id: highlightId,
          user_id: highlight.userId,
          analysis_type: "full"
        }
      );
      
      // Save results to Firestore
      await snap.ref.update({
        "aiAnalysis": {
          status: "completed",
          completedAt: Date.now(),
          metrics: response.data.metrics,
          raw_data: response.data.analysis
        }
      });
      
      return null;
    } catch (error) {
      console.error("Analysis failed:", error);
      await snap.ref.update({
        "aiAnalysis.status": "failed"
      });
      return null;
    }
  });
```

## Displaying Analysis Results

### Update Highlight Display

**In `app/highlights/[id]/page.tsx`:**

```typescript
{highlight.aiAnalysis?.status === "completed" && highlight.aiAnalysis.metrics && (
  <div className="mt-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-6">
    <h3 className="text-xl font-semibold text-[#0F172A] mb-4">
      Performance Metrics
    </h3>
    <div className="grid grid-cols-2 gap-4">
      {highlight.aiAnalysis.metrics.vertical_jump && (
        <div className="bg-white rounded-xl p-4">
          <p className="text-sm text-slate-600 mb-1">Vertical Jump</p>
          <p className="text-2xl font-bold text-[#3B82F6]">
            {highlight.aiAnalysis.metrics.vertical_jump}"
          </p>
        </div>
      )}
      {highlight.aiAnalysis.metrics.approach_speed && (
        <div className="bg-white rounded-xl p-4">
          <p className="text-sm text-slate-600 mb-1">Approach Speed</p>
          <p className="text-2xl font-bold text-[#3B82F6]">
            {highlight.aiAnalysis.metrics.approach_speed} m/s
          </p>
        </div>
      )}
      {highlight.aiAnalysis.metrics.action_type && (
        <div className="bg-white rounded-xl p-4">
          <p className="text-sm text-slate-600 mb-1">Action Type</p>
          <p className="text-2xl font-bold text-[#3B82F6] capitalize">
            {highlight.aiAnalysis.metrics.action_type}
          </p>
        </div>
      )}
      {highlight.aiAnalysis.metrics.kill_accuracy && (
        <div className="bg-white rounded-xl p-4">
          <p className="text-sm text-slate-600 mb-1">Kill Accuracy</p>
          <p className="text-2xl font-bold text-[#3B82F6]">
            {highlight.aiAnalysis.metrics.kill_accuracy}%
          </p>
        </div>
      )}
    </div>
  </div>
)}
```

### Add Analysis Status Indicator

```typescript
{highlight.aiAnalysis && (
  <div className="mt-2">
    {highlight.aiAnalysis.status === "pending" && (
      <span className="text-xs text-slate-500">Analysis pending...</span>
    )}
    {highlight.aiAnalysis.status === "processing" && (
      <span className="text-xs text-blue-500">Analyzing video...</span>
    )}
    {highlight.aiAnalysis.status === "failed" && (
      <span className="text-xs text-red-500">Analysis failed</span>
    )}
  </div>
)}
```

## API Reference

### POST `/analyze/video`

Analyze a single video.

**Request:**
```json
{
  "video_url": "https://res.cloudinary.com/.../video.mp4",
  "highlight_id": "abc123",
  "user_id": "user456",
  "analysis_type": "full"
}
```

**Response:**
```json
{
  "status": "success",
  "highlight_id": "abc123",
  "user_id": "user456",
  "analysis": {
    "frames_analyzed": 50,
    "player_detected": true,
    "ball_detected": true,
    "action_type": "kill"
  },
  "metrics": {
    "vertical_jump": 28.5,
    "approach_speed": 3.2,
    "ball_touch_detected": true,
    "kill_accuracy": 75.0,
    "max_reach_height": 28.5,
    "action_type": "kill"
  }
}
```

## Error Handling

The AI service may fail due to:
- Video format issues
- Network problems
- Model loading failures
- Processing timeouts

Always handle errors gracefully and provide user feedback:

```typescript
try {
  const result = await analyzeVideo(highlightId, videoUrl, userId);
  // Handle success
} catch (error) {
  if (error.response?.status === 500) {
    // Server error - retry later
  } else if (error.response?.status === 400) {
    // Invalid video - show user message
  } else {
    // Network error - check connection
  }
}
```

## Performance Considerations

- Video analysis can take 30-60 seconds for a 30-second clip
- Use async processing to avoid blocking the UI
- Consider queueing analysis jobs for high traffic
- Cache results to avoid re-analyzing the same video

## Security

- Add authentication to the AI service endpoint
- Validate video URLs to prevent SSRF attacks
- Rate limit analysis requests per user
- Use environment variables for service URLs

## Troubleshooting

### Analysis always fails
- Check AI service is running: `curl http://localhost:8000/health`
- Verify video URL is accessible
- Check service logs for errors

### Results are inaccurate
- Ensure SAM3 model weights are properly loaded
- Verify video quality and frame rate
- Check camera angle and lighting in videos

### Slow processing
- Reduce `max_frames` in analysis
- Use "quick" analysis type for faster results
- Consider GPU acceleration

