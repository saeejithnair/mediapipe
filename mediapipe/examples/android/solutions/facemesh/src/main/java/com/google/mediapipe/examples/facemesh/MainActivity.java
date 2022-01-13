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

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.opengl.GLES20;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.exifinterface.media.ExifInterface;
// ContentResolver dependency
import com.google.common.graph.Graph;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutioncore.CameraInput;
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView;
import com.google.mediapipe.solutioncore.VideoInput;
import com.google.mediapipe.solutions.facemesh.FaceMesh;
import com.google.mediapipe.solutions.facemesh.FaceMeshOptions;
import com.google.mediapipe.solutions.facemesh.FaceMeshResult;
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.GridLabelRenderer;
import com.jjoe64.graphview.LegendRenderer;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Main activity of MediaPipe Face Mesh app. */
public class MainActivity extends AppCompatActivity {
  private static final String TAG = "MainActivity";

  private FaceMesh facemesh;
  // Run the pipeline and the model inference on GPU or CPU.
  private static final boolean RUN_ON_GPU = false;

  private enum InputSource {
    UNKNOWN,
    IMAGE,
    VIDEO,
    CAMERA,
  }
  private InputSource inputSource = InputSource.UNKNOWN;
  // Image demo UI and image loader components.
  private ActivityResultLauncher<Intent> imageGetter;
  private FaceMeshResultImageView imageView;
  // Video demo UI and video loader components.
  private VideoInput videoInput;
  private ActivityResultLauncher<Intent> videoGetter;
  // Live camera demo UI and camera components.
  private CameraInput cameraInput;

  private SolutionGlSurfaceView<FaceMeshResult> glSurfaceView;

//  private LineGraphSeries<DataPoint> gRadiiRcheek = new LineGraphSeries<>();
//  private LineGraphSeries<DataPoint> gRadiiLcheek = new LineGraphSeries<>();
//  private LineGraphSeries<DataPoint> gRadiiForehead = new LineGraphSeries<>();

  private LineGraphSeries<DataPoint> gGreenIntensityForehead = new LineGraphSeries<>(new DataPoint[] {
          new DataPoint(0, 0)});
  private LineGraphSeries<DataPoint> gGreenIntensityRcheek = new LineGraphSeries<>(new DataPoint[] {
          new DataPoint(0, 0)});
  private LineGraphSeries<DataPoint> gGreenIntensityLcheek = new LineGraphSeries<>(new DataPoint[] {
          new DataPoint(0, 0)});

  private static int num_frames_counter = 1;

  private static final int[] TRACKERS_FOREHEAD = {104, 69, 108, 151, 337, 299, 333};
  private static final int[] TRACKERS_RCHEEK = {255, 261, 340, 352, 411, 427, 436};
  private static final int[] TRACKERS_LCHEEK = {25, 31, 111, 123, 187, 207, 216};
  private static final int FACE_LANDMARK_EXTREME_LEFT = 234;
  private static final int FACE_LANDMARK_EXTREME_RIGHT = 454;

  private static SplineInterpolator gInterpolator = new SplineInterpolator();

//  private final Handler mHandler = new Handler();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    // TODO: Add a toggle to switch between the original face mesh and attention mesh.
    setupStaticImageDemoUiComponents();
    setupVideoDemoUiComponents();
    setupLiveDemoUiComponents();
  }

  @Override
  protected void onResume() {
    super.onResume();
    if (inputSource == InputSource.CAMERA) {
      // Restarts the camera and the opengl surface rendering.
      cameraInput = new CameraInput(this);
      cameraInput.setNewFrameListener(textureFrame -> facemesh.send(textureFrame));
      glSurfaceView.post(this::startCamera);
      glSurfaceView.setVisibility(View.VISIBLE);
    } else if (inputSource == InputSource.VIDEO) {
      videoInput.resume();
    }
  }

  @Override
  protected void onPause() {
    super.onPause();
    if (inputSource == InputSource.CAMERA) {
      glSurfaceView.setVisibility(View.GONE);
      cameraInput.close();
    } else if (inputSource == InputSource.VIDEO) {
      videoInput.pause();
    }
  }

  private Bitmap downscaleBitmap(Bitmap originalBitmap) {
    double aspectRatio = (double) originalBitmap.getWidth() / originalBitmap.getHeight();
    int width = imageView.getWidth();
    int height = imageView.getHeight();
    if (((double) imageView.getWidth() / imageView.getHeight()) > aspectRatio) {
      width = (int) (height * aspectRatio);
    } else {
      height = (int) (width / aspectRatio);
    }
    return Bitmap.createScaledBitmap(originalBitmap, width, height, false);
  }

  private Bitmap rotateBitmap(Bitmap inputBitmap, InputStream imageData) throws IOException {
    int orientation =
        new ExifInterface(imageData)
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
    if (orientation == ExifInterface.ORIENTATION_NORMAL) {
      return inputBitmap;
    }
    Matrix matrix = new Matrix();
    switch (orientation) {
      case ExifInterface.ORIENTATION_ROTATE_90:
        matrix.postRotate(90);
        break;
      case ExifInterface.ORIENTATION_ROTATE_180:
        matrix.postRotate(180);
        break;
      case ExifInterface.ORIENTATION_ROTATE_270:
        matrix.postRotate(270);
        break;
      default:
        matrix.postRotate(0);
    }
    return Bitmap.createBitmap(
        inputBitmap, 0, 0, inputBitmap.getWidth(), inputBitmap.getHeight(), matrix, true);
  }

  /** Sets up the UI components for the static image demo. */
  private void setupStaticImageDemoUiComponents() {
    // The Intent to access gallery and read images as bitmap.
    imageGetter =
        registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
              Intent resultIntent = result.getData();
              if (resultIntent != null) {
                if (result.getResultCode() == RESULT_OK) {
                  Bitmap bitmap = null;
                  try {
                    bitmap =
                        downscaleBitmap(
                            MediaStore.Images.Media.getBitmap(
                                this.getContentResolver(), resultIntent.getData()));
                  } catch (IOException e) {
                    Log.e(TAG, "Bitmap reading error:" + e);
                  }
                  try {
                    InputStream imageData =
                        this.getContentResolver().openInputStream(resultIntent.getData());
                    bitmap = rotateBitmap(bitmap, imageData);
                  } catch (IOException e) {
                    Log.e(TAG, "Bitmap rotation error:" + e);
                  }
                  if (bitmap != null) {
                    facemesh.send(bitmap);
                  }
                }
              }
            });
    Button loadImageButton = findViewById(R.id.button_load_picture);
    loadImageButton.setOnClickListener(
        v -> {
          if (inputSource != InputSource.IMAGE) {
            stopCurrentPipeline();
            setupStaticImageModePipeline();
          }
          // Reads images from gallery.
          Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
          pickImageIntent.setDataAndType(MediaStore.Images.Media.INTERNAL_CONTENT_URI, "image/*");
          imageGetter.launch(pickImageIntent);
        });
    imageView = new FaceMeshResultImageView(this);
  }

  /** Sets up core workflow for static image mode. */
  private void setupStaticImageModePipeline() {
    this.inputSource = InputSource.IMAGE;
    // Initializes a new MediaPipe Face Mesh solution instance in the static image mode.
    facemesh =
        new FaceMesh(
            this,
            FaceMeshOptions.builder()
                .setStaticImageMode(true)
                .setRefineLandmarks(true)
                .setRunOnGpu(RUN_ON_GPU)
                .build());

    // Connects MediaPipe Face Mesh solution to the user-defined FaceMeshResultImageView.
    facemesh.setResultListener(
        faceMeshResult -> {
          logNoseLandmark(faceMeshResult, /*showPixelValues=*/ true);
          imageView.setFaceMeshResult(faceMeshResult);
          runOnUiThread(() -> imageView.update());
        });
    facemesh.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Face Mesh error:" + message));

    // Updates the preview layout.
    FrameLayout frameLayout = findViewById(R.id.preview_display_layout);
    frameLayout.removeAllViewsInLayout();
    imageView.setImageDrawable(null);
    frameLayout.addView(imageView);
    imageView.setVisibility(View.VISIBLE);
  }

  /** Sets up the UI components for the video demo. */
  private void setupVideoDemoUiComponents() {
    // The Intent to access gallery and read a video file.
    videoGetter =
        registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
              Intent resultIntent = result.getData();
              if (resultIntent != null) {
                if (result.getResultCode() == RESULT_OK) {
                  glSurfaceView.post(
                      () ->
                          videoInput.start(
                              this,
                              resultIntent.getData(),
                              facemesh.getGlContext(),
                              glSurfaceView.getWidth(),
                              glSurfaceView.getHeight()));
                }
              }
            });
    Button loadVideoButton = findViewById(R.id.button_load_video);
    loadVideoButton.setOnClickListener(
        v -> {
          stopCurrentPipeline();
          setupStreamingModePipeline(InputSource.VIDEO);
          // Reads video from gallery.
          Intent pickVideoIntent = new Intent(Intent.ACTION_PICK);
          pickVideoIntent.setDataAndType(MediaStore.Video.Media.INTERNAL_CONTENT_URI, "video/*");
          videoGetter.launch(pickVideoIntent);
        });
  }

  /** Sets up the UI components for the live demo with camera input. */
  private void setupLiveDemoUiComponents() {
    Button startCameraButton = findViewById(R.id.button_start_camera);
    startCameraButton.setOnClickListener(
        v -> {
          if (inputSource == InputSource.CAMERA) {
            return;
          }
          stopCurrentPipeline();
          setupStreamingModePipeline(InputSource.CAMERA);
        });
  }

  /** Sets up core workflow for streaming mode. */
  private void setupStreamingModePipeline(InputSource inputSource) {
    this.inputSource = inputSource;
    // Initializes a new MediaPipe Face Mesh solution instance in the streaming mode.
    facemesh =
        new FaceMesh(
            this,
            FaceMeshOptions.builder()
                .setStaticImageMode(false)
                .setRefineLandmarks(true)
                .setRunOnGpu(RUN_ON_GPU)
                .build());
    facemesh.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Face Mesh error:" + message));

    if (inputSource == InputSource.CAMERA) {
      cameraInput = new CameraInput(this);
      cameraInput.setNewFrameListener(textureFrame -> facemesh.send(textureFrame));
    } else if (inputSource == InputSource.VIDEO) {
      videoInput = new VideoInput(this);
      videoInput.setNewFrameListener(textureFrame -> facemesh.send(textureFrame));
    }

    GraphView graph = (GraphView) findViewById(R.id.graph);
    assert graph != null;

//    graph.addSeries(gRadiiForehead);
//    graph.addSeries(gRadiiLcheek);
//    graph.addSeries(gRadiiRcheek);
    //    gRadiiForehead.setTitle("Forehead");
//    gRadiiForehead.setColor(0xFFFF0000);
//    gRadiiLcheek.setTitle("LCheek");
//    gRadiiLcheek.setColor(0xFF00FF00);
//    gRadiiRcheek.setTitle("RCheek");
//    gRadiiRcheek.setColor(0xFF0000FF);


    graph.addSeries(gGreenIntensityForehead);
    graph.addSeries(gGreenIntensityLcheek);
    graph.addSeries(gGreenIntensityRcheek);

    gGreenIntensityForehead.setTitle("Forehead");
    gGreenIntensityForehead.setColor(0xFF0000FF);
    gGreenIntensityLcheek.setTitle("LCheek");
    gGreenIntensityLcheek.setColor(0xFF00FF00);
    gGreenIntensityRcheek.setTitle("RCheek");
    gGreenIntensityRcheek.setColor(0xFFFF0000);


    graph.getViewport().setXAxisBoundsManual(true);
    graph.getViewport().setYAxisBoundsManual(true);
    graph.getViewport().setMinX(0);
    graph.getViewport().setMaxX(40);
    graph.getViewport().setMinY(0);
    graph.getViewport().setMaxY(255);

    graph.getLegendRenderer().setVisible(true);
    graph.getLegendRenderer().setAlign(LegendRenderer.LegendAlign.TOP);

    GridLabelRenderer gridLabel = graph.getGridLabelRenderer();
    gridLabel.setVerticalAxisTitle("Mean Intensity (Green)");
    gridLabel.setHorizontalAxisTitle("Frame Number");


    // Initializes a new Gl surface view with a user-defined FaceMeshResultGlRenderer.
    glSurfaceView =
        new SolutionGlSurfaceView<>(this, facemesh.getGlContext(), facemesh.getGlMajorVersion());
    glSurfaceView.setSolutionResultRenderer(new FaceMeshResultGlRenderer());
    glSurfaceView.setRenderInputImage(true);
    facemesh.setResultListener(
        faceMeshResult -> {
          logNoseLandmark(faceMeshResult, /*showPixelValues=*/ false);
//          updateRadiusGraph(faceMeshResult, num_frames_counter, gRadiiForehead, gRadiiLcheek, gRadiiRcheek);
          updateIntensityGraph(faceMeshResult, num_frames_counter, gGreenIntensityForehead, gGreenIntensityRcheek, gGreenIntensityLcheek);
          num_frames_counter++;
          glSurfaceView.setRenderData(faceMeshResult);
          glSurfaceView.requestRender();
        });

    // The runnable to start camera after the gl surface view is attached.
    // For video input source, videoInput.start() will be called when the video uri is available.
    if (inputSource == InputSource.CAMERA) {
      glSurfaceView.post(this::startCamera);
    }

    graph.onDataChanged(true, true);

    // Updates the preview layout.
    FrameLayout frameLayout = findViewById(R.id.preview_display_layout);
    imageView.setVisibility(View.GONE);
    frameLayout.removeAllViewsInLayout();
    frameLayout.addView(glSurfaceView);
    glSurfaceView.setVisibility(View.VISIBLE);
    frameLayout.requestLayout();
  }

  private void startCamera() {
    cameraInput.start(
        this,
        facemesh.getGlContext(),
        CameraInput.CameraFacing.FRONT,
        glSurfaceView.getWidth(),
        glSurfaceView.getHeight());
  }

  private void stopCurrentPipeline() {
    if (cameraInput != null) {
      cameraInput.setNewFrameListener(null);
      cameraInput.close();
    }
    if (videoInput != null) {
      videoInput.setNewFrameListener(null);
      videoInput.close();
    }
    if (glSurfaceView != null) {
      glSurfaceView.setVisibility(View.GONE);
    }
    if (facemesh != null) {
      facemesh.close();
    }
  }

  private void logNoseLandmark(FaceMeshResult result, boolean showPixelValues) {
    if (result == null || result.multiFaceLandmarks().isEmpty()) {
      return;
    }
    NormalizedLandmark noseLandmark = result.multiFaceLandmarks().get(0).getLandmarkList().get(1);
    // For Bitmaps, show the pixel values. For texture inputs, show the normalized coordinates.
    if (showPixelValues) {
      int width = result.inputBitmap().getWidth();
      int height = result.inputBitmap().getHeight();
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Mesh nose coordinates (pixel values): x=%f, y=%f",
              noseLandmark.getX() * width, noseLandmark.getY() * height));
    } else {
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Mesh nose normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              noseLandmark.getX(), noseLandmark.getY()));
    }
  }

  private void updateIntensityGraph(FaceMeshResult result, int num_frames_counter,
                                    LineGraphSeries<DataPoint> green_forehead,
                                    LineGraphSeries<DataPoint> green_rcheek,
                                    LineGraphSeries<DataPoint> green_lcheek) {
    int numFaces = result.multiFaceLandmarks().size();
    for (int i=0; i<numFaces; ++i) {
      List<NormalizedLandmark> faceLandmarkList = result.multiFaceLandmarks().get(i).getLandmarkList();

      double mean_intensity_forehead = calculateMeanIntensityGreen(result.inputBitmap(),
              faceLandmarkList, FaceMeshResultGlRenderer.LANDMARKS_FOREHEAD);
      double mean_intensity_lcheek = calculateMeanIntensityGreen(result.inputBitmap(),
              faceLandmarkList, FaceMeshResultGlRenderer.LANDMARKS_LCHEEK);
      double mean_intensity_rcheek = calculateMeanIntensityGreen(result.inputBitmap(),
              faceLandmarkList, FaceMeshResultGlRenderer.LANDMARKS_RCHEEK);

      green_forehead.appendData(new DataPoint(num_frames_counter, mean_intensity_forehead), true, 40);
      green_rcheek.appendData(new DataPoint(num_frames_counter, mean_intensity_rcheek), true, 40);
      green_lcheek.appendData(new DataPoint(num_frames_counter, mean_intensity_lcheek), true, 40);
    }
  }

  private double calculateMeanIntensityGreen(Bitmap bmp,
                                             List<NormalizedLandmark> faceLandmarkList,
                                             int[] landmark_indices) {
    int cum_sum = 0;
    int width = bmp.getWidth();
    int height = bmp.getHeight();
    ConvexHull.Point point;

    List<ConvexHull.Point> hull_points = FaceMeshResultGlRenderer.calculateContour(faceLandmarkList,
            landmark_indices);

    for (int i=0; i<hull_points.size(); ++i) {
      // Get landmark in contour region.
//      NormalizedLandmark landmark = faceLandmarkList.get(landmark_indices[i]);
      point = hull_points.get(i);
      // Convert normalized coordinate to pixels.
      double px_x = point.x * width;
      double px_y = (1-point.y) * height;

      // Get pixel intensity and add to cumulative sum.
      int pixel = bmp.getPixel((int)px_x, (int)px_y);
      cum_sum += Color.green(pixel);
    }

    double mean_intensity = cum_sum / hull_points.size();

    return mean_intensity;
  }

  private void updateRadiusGraph(FaceMeshResult result, int num_frames_counter,
                                 LineGraphSeries<DataPoint> radii_forehead,
                                 LineGraphSeries<DataPoint> radii_lcheek,
                                 LineGraphSeries<DataPoint> radii_rcheek) {

    int numFaces = result.multiFaceLandmarks().size();
    for (int i = 0; i < numFaces; ++i) {
      double[] forehead_tracker_coords_x = new double[TRACKERS_FOREHEAD.length];
      double[] rcheek_tracker_coords_x = new double[TRACKERS_RCHEEK.length];
      double[] lcheek_tracker_coords_x = new double[TRACKERS_LCHEEK.length];
      double[] forehead_tracker_coords_y = new double[TRACKERS_FOREHEAD.length];
      double[] rcheek_tracker_coords_y = new double[TRACKERS_RCHEEK.length];
      double[] lcheek_tracker_coords_y = new double[TRACKERS_LCHEEK.length];

      double face_width = calculateFaceWidth(result.multiFaceLandmarks().get(i).getLandmarkList());
      List<NormalizedLandmark> faceLandmarkList = result.multiFaceLandmarks().get(i).getLandmarkList();

      double radius_forehead = calculateRadius(faceLandmarkList, TRACKERS_FOREHEAD,
              forehead_tracker_coords_x, forehead_tracker_coords_y, face_width, true);
      double radius_lcheek = calculateRadius(faceLandmarkList, TRACKERS_LCHEEK, lcheek_tracker_coords_x,
              lcheek_tracker_coords_y, face_width, false);
      double radius_rcheek = calculateRadius(faceLandmarkList, TRACKERS_RCHEEK, rcheek_tracker_coords_x,
              rcheek_tracker_coords_y, face_width,true);

      radii_forehead.appendData(new DataPoint(num_frames_counter, radius_forehead), true, 40);
      radii_lcheek.appendData(new DataPoint(num_frames_counter, radius_lcheek), true, 40);
      radii_rcheek.appendData(new DataPoint(num_frames_counter, radius_rcheek), true, 40);
    }
  }

  private void moveToOrigin(double[] x_coords, double[] y_coords, boolean flip_y_coords) {
    double x0 = x_coords[0];
    double y0 = y_coords[0];

    if (flip_y_coords) {
      y0 = -y0;
    }

    assert x_coords.length == y_coords.length;

    for (int i=0; i<x_coords.length; ++i) {
      if (flip_y_coords) {
        y_coords[i] = -y_coords[i] - y0;
      } else {
        y_coords[i] = y_coords[i] - y0;
      }
      x_coords[i] = x_coords[i] - x0;
    }
  }

  private void rotateToXAxis(double[] x_coords, double[] y_coords) {
    double xn = x_coords[x_coords.length-1];
    double yn = y_coords[y_coords.length-1];

    assert x_coords.length == y_coords.length;

    double theta = Math.atan2(-yn, xn);

    for (int i=0; i<x_coords.length; ++i) {
      double cur_x = x_coords[i];
      double cur_y = y_coords[i];

      x_coords[i] = cur_x * Math.cos(theta) - cur_y * Math.sin(theta);
      y_coords[i] = cur_x * Math.sin(theta) + cur_y * Math.cos(theta);
    }
  }

  private double getRad(double[] x_coords, double[] y_coords) {
    UnivariateDifferentiableFunction y_spl = gInterpolator.interpolate(x_coords, y_coords);
    double midpoint = (x_coords[0] + x_coords[x_coords.length-1])/2;
    DerivativeStructure y_spl_d2 = y_spl.value(new DerivativeStructure(1, 2, 0, midpoint));

    // Calculate radius of curvature.
    return (1/Math.abs(y_spl_d2.getPartialDerivative(2)));
  }

  private double calculateRadius(List<NormalizedLandmark> faceLandmarkList, int[] tracker_indices,
                               double[] tracker_coords_x, double[] tracker_coords_y,
                               double face_width, boolean flip_y_coords) {
    assert tracker_coords_x.length == tracker_indices.length;
    assert tracker_coords_y.length == tracker_indices.length;

    List<NormalizedLandmark> landmarks = new ArrayList<NormalizedLandmark>();

    for (int i=0; i<tracker_indices.length; ++i) {
      NormalizedLandmark landmark = faceLandmarkList.get(tracker_indices[i]);
      landmarks.add(landmark);
    }

    Collections.sort(landmarks, new FaceMeshResultGlRenderer.NormalizedLandmarkCompare());

    for (int i=0; i<tracker_indices.length; ++i) {
      NormalizedLandmark landmark = landmarks.get(i);
      tracker_coords_x[i] = landmark.getX();
      tracker_coords_y[i] = landmark.getY();
    }

    // Rotate points so that we can use the x-axis as the reference baseline for getting the max
    // value.
    rotateToXAxis(tracker_coords_x, tracker_coords_y);

    // Sort the points by x-coordinate so that the list is monotonic
    List<ConvexHull.Point> tracker_coords = new ArrayList<>();
    for (int i=0; i<tracker_indices.length; ++i) {
      tracker_coords.add(new ConvexHull.Point(tracker_coords_x[i], tracker_coords_y[i]));
    }
    Collections.sort(tracker_coords, new FaceMeshResultGlRenderer.PointCompare());

    // Dump list of points into two separate arrays (since spline interpolation can only be done
    // on using two distinct arrays, not objects.
    convertPointsListToArrays(tracker_coords, tracker_coords_x, tracker_coords_y);

    // Translate all points such that the first point is at the origin
    moveToOrigin(tracker_coords_x, tracker_coords_y, flip_y_coords);

    // Calculate radius.
    return (getRad(tracker_coords_x, tracker_coords_y) / face_width * 10);
  }

  private double calculateFaceWidth(List<NormalizedLandmark> faceLandmarkList) {
    NormalizedLandmark face_left = faceLandmarkList.get(FACE_LANDMARK_EXTREME_LEFT);
    NormalizedLandmark face_right = faceLandmarkList.get(FACE_LANDMARK_EXTREME_RIGHT);

    double delta_x = face_left.getX() - face_right.getX();
    double delta_y = face_left.getY() - face_right.getY();

    // Calculate euclidean distance between two points.
    return Math.sqrt(delta_x*delta_x + delta_y*delta_y);
  }

  private void convertPointsListToArrays(List<ConvexHull.Point> points, double[] points_x, double[] points_y) {
    assert points_x.length == points_y.length;
    assert points_x.length == points.size();

    for (int i=0; i<points_x.length; ++i) {
      points_x[i] = points.get(i).x;
      points_y[i] = points.get(i).y;
    }
  }
}
