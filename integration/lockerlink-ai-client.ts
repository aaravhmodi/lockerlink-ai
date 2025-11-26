/**
 * LockerLink AI Service Client
 * 
 * Utility functions to interact with the LockerLink AI microservice
 * for volleyball video analysis.
 */

const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || "http://localhost:8000";

export interface VideoAnalysisRequest {
  video_url: string;
  highlight_id?: string;
  user_id?: string;
  analysis_type?: "full" | "quick" | "player_only" | "ball_only";
}

export interface VideoAnalysisResponse {
  status: "success" | "error";
  highlight_id?: string;
  user_id?: string;
  analysis?: {
    frames_analyzed: number;
    player_detected: boolean;
    ball_detected: boolean;
    action_type: "kill" | "block" | "other" | "unknown";
  };
  metrics?: {
    vertical_jump?: number;
    approach_speed?: number;
    ball_touch_detected: boolean;
    kill_accuracy?: number;
    max_reach_height?: number;
    contact_point?: number;
    action_type: string;
  };
  raw_data?: {
    player_tracks: any[];
    ball_tracks: any[];
    key_frames: any[];
  };
  error?: string;
}

/**
 * Analyze a volleyball highlight video
 * 
 * @param highlightId - Firestore highlight document ID
 * @param videoUrl - Cloudinary video URL
 * @param userId - User ID who owns the highlight
 * @param analysisType - Type of analysis to perform
 * @returns Analysis results
 */
export async function analyzeVideo(
  highlightId: string,
  videoUrl: string,
  userId?: string,
  analysisType: "full" | "quick" | "player_only" | "ball_only" = "full"
): Promise<VideoAnalysisResponse> {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/analyze/video`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_url: videoUrl,
        highlight_id: highlightId,
        user_id: userId,
        analysis_type: analysisType,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data as VideoAnalysisResponse;
  } catch (error: any) {
    console.error("Error analyzing video:", error);
    return {
      status: "error",
      error: error.message || "Failed to analyze video",
    };
  }
}

/**
 * Check if AI service is healthy
 */
export async function checkAIServiceHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/health`);
    if (!response.ok) return false;
    const data = await response.json();
    return data.status === "healthy" && data.model_loaded === true;
  } catch (error) {
    console.error("AI service health check failed:", error);
    return false;
  }
}

/**
 * Analyze multiple videos in batch
 * 
 * @param videoUrls - Array of video URLs to analyze
 * @returns Array of analysis results
 */
export async function analyzeBatchVideos(
  videoUrls: string[]
): Promise<VideoAnalysisResponse[]> {
  try {
    const response = await fetch(`${AI_SERVICE_URL}/analyze/batch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(videoUrls),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.results || [];
  } catch (error: any) {
    console.error("Error in batch analysis:", error);
    return videoUrls.map(() => ({
      status: "error" as const,
      error: error.message || "Batch analysis failed",
    }));
  }
}

