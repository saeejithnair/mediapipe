// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.examples.facemesh;

import android.opengl.GLES20;
import com.google.common.collect.ImmutableSet;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.facemesh.FaceMesh;
import com.google.mediapipe.solutions.facemesh.FaceMeshConnections;
import com.google.mediapipe.solutions.facemesh.FaceMeshResult;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/** A custom implementation of {@link ResultGlRenderer} to render {@link FaceMeshResult}. */
public class FaceMeshResultGlRenderer implements ResultGlRenderer<FaceMeshResult> {
  private static final String TAG = "FaceMeshResultGlRenderer";

  private static final float[] TESSELATION_COLOR = new float[] {0.75f, 0.75f, 0.75f, 0.5f};
  private static final int TESSELATION_THICKNESS = 5;
  private static final float[] RIGHT_EYE_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final int RIGHT_EYE_THICKNESS = 8;
  private static final float[] RIGHT_EYEBROW_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final int RIGHT_EYEBROW_THICKNESS = 8;
  private static final float[] LEFT_EYE_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final int LEFT_EYE_THICKNESS = 8;
  private static final float[] LEFT_EYEBROW_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final int LEFT_EYEBROW_THICKNESS = 8;
  private static final float[] FACE_OVAL_COLOR = new float[] {0.9f, 0.9f, 0.9f, 1f};
  private static final int FACE_OVAL_THICKNESS = 8;
  private static final float[] LIPS_COLOR = new float[] {0.9f, 0.9f, 0.9f, 1f};
  private static final int LIPS_THICKNESS = 8;
  private static final String VERTEX_SHADER =
      "uniform mat4 uProjectionMatrix;\n"
          + "attribute vec4 vPosition;\n"
          + "void main() {\n"
          + "  gl_Position = uProjectionMatrix * vPosition;\n"
          + "}";
  private static final String FRAGMENT_SHADER =
      "precision mediump float;\n"
          + "uniform vec4 uColor;\n"
          + "void main() {\n"
          + "  gl_FragColor = uColor;\n"
          + "}";

  // Landmark keypoint indices for the facial features we're interested in.
  // Each value in a LANDMARKS* list corresponds to a point on the mesh for that facial area.
  private static final int[] LANDMARKS_LCHEEK = {454,447,345,346,347,348,349,350,357,343,437,420,279,358,423,426,436,432,430,394,379,365,397,288,361,323,366,352,280,330,425,411,376,416,367,435,401};
  private static final int[] LANDMARKS_FOREHEAD = {127,34,162,21,139,143,35,156,71,54,103,68,70,124,226,130,113,46,63,104,67,69,105,53,52,66,108,109,10,151,107,9,55,65,8,285,336,337,338,297,299,296,295,282,334,333,332,284,298,293,283,276,300,301,251,389,368,383,353,356,264,372,265};
  private static final int[] LANDMARKS_RCHEEK = {234,227,116,117,118,119,120,121,128,114,217,198,49,129,203,206,216,214,135,136,172,58,132,93,47,126,209,142,100,101,36,207,205,50,123,137,177,147,177,213,215,192,138};
  private static final int[] TRACKERS_FOREHEAD = {104, 69, 108, 151, 337, 299, 333};
  private static final int[] TRACKERS_RCHEEK = {255, 261, 340, 352, 411, 427, 436};
  private static final int[] TRACKERS_LCHEEK = {25, 31, 111, 123, 187, 207, 216};
  private static final int FACE_LANDMARK_EXTREME_LEFT = 234;
  private static final int FACE_LANDMARK_EXTREME_RIGHT = 454;

  private int program;
  private int positionHandle;
  private int projectionMatrixHandle;
  private int colorHandle;

  private int loadShader(int type, String shaderCode) {
    int shader = GLES20.glCreateShader(type);
    GLES20.glShaderSource(shader, shaderCode);
    GLES20.glCompileShader(shader);
    return shader;
  }

  @Override
  public void setupRendering() {
    program = GLES20.glCreateProgram();
    int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
    int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    GLES20.glAttachShader(program, vertexShader);
    GLES20.glAttachShader(program, fragmentShader);
    GLES20.glLinkProgram(program);
    positionHandle = GLES20.glGetAttribLocation(program, "vPosition");
    projectionMatrixHandle = GLES20.glGetUniformLocation(program, "uProjectionMatrix");
    colorHandle = GLES20.glGetUniformLocation(program, "uColor");
  }

  @Override
  public void renderResult(FaceMeshResult result, float[] projectionMatrix) {
    if (result == null) {
      return;
    }
    GLES20.glUseProgram(program);
    GLES20.glUniformMatrix4fv(projectionMatrixHandle, 1, false, projectionMatrix, 0);

    int numFaces = result.multiFaceLandmarks().size();
    for (int i = 0; i < numFaces; ++i) {
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_TESSELATION,
          TESSELATION_COLOR,
          TESSELATION_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_RIGHT_EYE,
          RIGHT_EYE_COLOR,
          RIGHT_EYE_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_RIGHT_EYEBROW,
          RIGHT_EYEBROW_COLOR,
          RIGHT_EYEBROW_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LEFT_EYE,
          LEFT_EYE_COLOR,
          LEFT_EYE_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LEFT_EYEBROW,
          LEFT_EYEBROW_COLOR,
          LEFT_EYEBROW_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_FACE_OVAL,
          FACE_OVAL_COLOR,
          FACE_OVAL_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          FaceMeshConnections.FACEMESH_LIPS,
          LIPS_COLOR,
          LIPS_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          LANDMARKS_RCHEEK,
          RIGHT_EYE_COLOR,
          RIGHT_EYE_THICKNESS);
      drawLandmarks(
          result.multiFaceLandmarks().get(i).getLandmarkList(),
          LANDMARKS_LCHEEK,
          LEFT_EYE_COLOR,
          LEFT_EYE_THICKNESS);
      if (result.multiFaceLandmarks().get(i).getLandmarkCount()
          == FaceMesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES) {
        drawLandmarks(
            result.multiFaceLandmarks().get(i).getLandmarkList(),
            FaceMeshConnections.FACEMESH_RIGHT_IRIS,
            RIGHT_EYE_COLOR,
            RIGHT_EYE_THICKNESS);
        drawLandmarks(
            result.multiFaceLandmarks().get(i).getLandmarkList(),
            FaceMeshConnections.FACEMESH_LEFT_IRIS,
            LEFT_EYE_COLOR,
            LEFT_EYE_THICKNESS);
      }
    }
  }

  /**
   * Deletes the shader program.
   *
   * <p>This is only necessary if one wants to release the program while keeping the context around.
   */
  public void release() {
    GLES20.glDeleteProgram(program);
  }

  private void drawLandmarks(
      List<NormalizedLandmark> faceLandmarkList,
      ImmutableSet<FaceMeshConnections.Connection> connections,
      float[] colorArray,
      int thickness) {
    GLES20.glUniform4fv(colorHandle, 1, colorArray, 0);
    GLES20.glLineWidth(thickness);
    for (FaceMeshConnections.Connection c : connections) {
      drawConnection(faceLandmarkList, c.start(), c.end());
    }
  }

  private void drawLandmarks(
          List<NormalizedLandmark> faceLandmarkList,
          int[] connection_landmarks,
          float[] colorArray,
          int thickness) {
    int list_size = connection_landmarks.length;
    assert list_size >= 2 : String.format("Received list of size %d, expected 2 or more elements.", list_size);

    GLES20.glUniform4fv(colorHandle, 1, colorArray, 0);
    GLES20.glLineWidth(thickness);

    calculateContour(faceLandmarkList, connection_landmarks);

//    // Draw line between consecutive landmark points.
//    for (int i=1; i<list_size; ++i) {
//      drawConnection(faceLandmarkList, connection_landmarks[i-1], connection_landmarks[i]);
//    }
  }

  private void calculateContour(List<NormalizedLandmark> faceLandmarkList,
                                int[] connection_landmarks) {
    List<ConvexHull.Point> points = new ArrayList<ConvexHull.Point>();

    ConvexHull convex_hull = new ConvexHull();

    for (int i=0; i<connection_landmarks.length; ++i) {
      NormalizedLandmark landmark = faceLandmarkList.get(connection_landmarks[i]);
      points.add(new ConvexHull.Point(landmark.getX(), landmark.getY()));
    }

    // Calculate convex hull from list of landmarks known to correspond to region.
    // E.g. create contour approximating cheek based on all landmarks known to be
    // in the right cheek region.
    List<ConvexHull.Point> hull_points = convex_hull.convexHull(points);

    int hull_points_size = hull_points.size();
    if (hull_points_size < 2) {
      // If less than two points, not a contour. Nothing to display.
      // TODO(snair): Log error/warning message?
      return;
    }

    for (int i=1; i<hull_points_size; ++i) {
      float[] vertex = {hull_points.get(i-1).x, hull_points.get(i-1).y,
              hull_points.get(i).x, hull_points.get(i).y};
      showVertex(vertex);
    }

    // Draw connection between first and last vertex.
    float[] vertex = {hull_points.get(0).x, hull_points.get(0).y,
            hull_points.get(hull_points_size-1).x, hull_points.get(hull_points_size-1).y};
    showVertex(vertex);
  }

  private void drawConnection(List<NormalizedLandmark> faceLandmarkList,
                              int connection_start, int connection_end) {
    NormalizedLandmark start = faceLandmarkList.get(connection_start);
    NormalizedLandmark end = faceLandmarkList.get(connection_end);
    float[] vertex = {start.getX(), start.getY(), end.getX(), end.getY()};
    showVertex(vertex);
  }

  private void showVertex(float[] vertex) {
    FloatBuffer vertexBuffer =
            ByteBuffer.allocateDirect(vertex.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
                    .put(vertex);
    vertexBuffer.position(0);
    GLES20.glEnableVertexAttribArray(positionHandle);
    GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
    GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2);
  }
}
