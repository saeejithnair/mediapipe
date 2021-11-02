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
import android.util.Log;

import com.google.common.collect.ImmutableSet;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.facemesh.FaceMesh;
import com.google.mediapipe.solutions.facemesh.FaceMeshConnections;
import com.google.mediapipe.solutions.facemesh.FaceMeshResult;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;

/** A custom implementation of {@link ResultGlRenderer} to render {@link FaceMeshResult}. */
public class FaceMeshResultGlRenderer implements ResultGlRenderer<FaceMeshResult> {

  public static class NormalizedLandmarkCompare implements Comparator<NormalizedLandmark> {
    @Override
    public int compare(NormalizedLandmark o1, NormalizedLandmark o2) {
      // Compare/sort based on x coordinates.
      return Float.compare(o1.getX(), o2.getX());
    }
  }

  public static class PointCompare implements Comparator<ConvexHull.Point> {
    @Override
    public int compare(ConvexHull.Point o1, ConvexHull.Point o2) {
      // Compare/sort based on x coordinates.
      return Float.compare(o1.x, o2.x);
    }
  }

  // Colours are in {R, G, B, alpha} format.
  private static final float[] TESSELATION_COLOR = new float[] {0.75f, 0.75f, 0.75f, 0.5f};
  private static final int TESSELATION_THICKNESS = 5;
  private static final float[] RIGHT_EYE_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final int RIGHT_EYE_THICKNESS = 8;
  private static final float[] RIGHT_CHEEK_COLOR = new float[] {1f, 0.2f, 0.2f, 1f};
  private static final int RIGHT_CHEEK_THICKNESS = 8;
  private static final float[] LEFT_CHEEK_COLOR = new float[] {0.2f, 1f, 0.2f, 1f};
  private static final int LEFT_CHEEK_THICKNESS = 8;
  private static final float[] FACE_OVAL_COLOR = new float[] {0.9f, 0.9f, 0.9f, 1f};
  private static final int FACE_OVAL_THICKNESS = 8;
  private static final float[] FOREHEAD_COLOR = new float[] {0.2f, 0.2f, 1f, 1f};
  private static final int FOREHEAD_THICKNESS = 8;
  private static final float[] TRACKER_COLOR = new float[] {1f, 1f, 0.2f, 1f};
  private static final int TRACKER_THICKNESS = 8;
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
  private static final int[] LANDMARKS_LCHEEK = {345, 340, 346, 347, 330, 266, 423, 426, 436, 427, 411, 376, 352, 280, 425};
  private static final int[] LANDMARKS_FOREHEAD = {103, 67,109, 10, 338, 297, 332, 334, 296, 336, 107, 66, 105, 104, 69, 108, 151, 337, 299, 333};
  private static final int[] LANDMARKS_RCHEEK = {116, 111, 117, 118, 101, 203, 206, 216, 207, 205, 187, 147, 123, 50};

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
      List<NormalizedLandmark> faceLandmarkList = result.multiFaceLandmarks().get(i).getLandmarkList();

      drawLandmarks(faceLandmarkList,
          FaceMeshConnections.FACEMESH_TESSELATION,
          TESSELATION_COLOR,
          TESSELATION_THICKNESS);
      drawLandmarks(faceLandmarkList,
          FaceMeshConnections.FACEMESH_FACE_OVAL,
          FACE_OVAL_COLOR,
          FACE_OVAL_THICKNESS);
      drawLandmarks(faceLandmarkList,
          LANDMARKS_RCHEEK,
          RIGHT_CHEEK_COLOR,
          RIGHT_CHEEK_THICKNESS);
      drawLandmarks(faceLandmarkList,
          LANDMARKS_LCHEEK,
          LEFT_CHEEK_COLOR,
          LEFT_CHEEK_THICKNESS);
      drawLandmarks(faceLandmarkList,
          LANDMARKS_FOREHEAD,
          FOREHEAD_COLOR,
          FOREHEAD_THICKNESS);
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

    List<ConvexHull.Point> hull_points = calculateContour(faceLandmarkList, connection_landmarks);

    // Draw contour created through convex hull.
    int hull_points_size = hull_points.size();
    if (hull_points_size < 2) {
      // If less than two points, not a contour. Nothing to display.
      Log.e("calculateContour", String.format("Invalid hull created with %d vertices", hull_points_size));
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

  private List<ConvexHull.Point> calculateContour(List<NormalizedLandmark> faceLandmarkList,
                                int[] connection_landmarks) {
    List<ConvexHull.Point> points = new ArrayList<>();

    for (int i=0; i<connection_landmarks.length; ++i) {
      NormalizedLandmark landmark = faceLandmarkList.get(connection_landmarks[i]);
      points.add(new ConvexHull.Point(landmark.getX(), landmark.getY()));
    }

    // Calculate convex hull from list of landmarks known to correspond to region.
    // E.g. create contour approximating cheek based on all landmarks known to be
    // in the right cheek region.
    return ConvexHull.convexHull(points);
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
