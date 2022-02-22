#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

#include "mediapipe/graphs/mobio/constants.h"

namespace mediapipe {

namespace {

constexpr char kImageTag[] = "IMAGE";
constexpr char kContourTag[] = "CONTOUR_INTENSITY_FOREHEAD";
constexpr char kIntensityTag[] = "INTENSITY";

constexpr uint8_t kColourInsideContour = 255;
constexpr uint8_t kColourOutsideContour = 0;

} // anonymous namespace

class MobioIntensityCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  // Translates points in place based on following transformation.
  // points[i] = points[i] + translation_offset
  absl::Status TranslatePoints(mobio::Points* points, 
                               const mobio::Point& translation_offset);

  // Calculates the mean intensity of pixels within contour region inside image.
  // Pixel is inside contour if mask[i,j] has non-zero value.
  absl::Status ComputeMeanIntensityContour(
                      const cv::Mat& image,
                      const cv::Mat& mask,
                      std::unique_ptr<mobio::Intensity>& mean_intensity);

};

REGISTER_CALCULATOR(MobioIntensityCalculator);

absl::Status MobioIntensityCalculator::GetContract(CalculatorContract* cc) {
  // Validate input/output stream tags connected to node.
  RET_CHECK(cc->Inputs().HasTag(kImageTag));
  RET_CHECK(cc->Inputs().HasTag(kContourTag));
  RET_CHECK(cc->Outputs().HasTag(kIntensityTag));
  
  // Set type for input stream packets.
  cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Inputs().Tag(kContourTag).Set<LandmarkList>();

  // Set type for output stream packets.
  cc->Outputs().Tag(kIntensityTag).Set<mobio::Intensity>();

  return absl::OkStatus();
}

absl::Status MobioIntensityCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}

absl::Status MobioIntensityCalculator::TranslatePoints(
                mobio::Points* points, const mobio::Point& translation_offset) {
  RET_CHECK(points != nullptr) << "Failed nullptr check for points container.";

  for (auto& point: *points) {
    point += translation_offset;
  }

  return absl::OkStatus();
}

absl::Status MobioIntensityCalculator::ComputeMeanIntensityContour(
                const cv::Mat& image,
                const cv::Mat& mask,
                std::unique_ptr<mobio::Intensity>& mean_intensity) {
  // Calculate mean intensity of pixels within the contour.
  const cv::Scalar mean_intensity_scalar = cv::mean(image, mask);

  // Copy values from cv::Scalar to mobio::Intensity (output format).
  mean_intensity = absl::make_unique<mobio::Intensity>(mean_intensity_scalar[0], 
                            mean_intensity_scalar[1], mean_intensity_scalar[2]);
  return absl::OkStatus();
}

absl::Status MobioIntensityCalculator::Process(CalculatorContext* cc) {
  LOG(INFO) << "In MobioIntensityCalculator Process()";
  if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().Tag(kContourTag).IsEmpty()) {
    return absl::OkStatus();
  }

  LOG(INFO) << "Getting input frame";

  const auto& input_frame = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
  const auto input_frame_width = input_frame.Width();
  const auto input_frame_height = input_frame.Height();

  LOG(INFO) << "Getting input contour";

  // Get vector of points enclosing contour region.
  const LandmarkList contour_landmark_points = cc->Inputs().Tag(kContourTag).Get<LandmarkList>();

  mobio::Points contour_points;
  contour_points.reserve(contour_landmark_points.landmark_size());
  for (int i=0; i<contour_landmark_points.landmark_size(); ++i) {
    const auto& landmark_point = contour_landmark_points.landmark(i);
    contour_points.emplace_back(mobio::Point(landmark_point.x(), landmark_point.y()));
  }

  LOG(INFO) << "Logging contour points.";
  for (auto& point:contour_points) {
    LOG(INFO) << "x: " << point.x << ", y: " << point.y;
  }

  LOG(INFO) << "Drawing convex hull";
  // Compute convex hull for each region's contour.
  // This is basically a more tightly wrapped contour.
  mobio::Points hull_points;
  cv::convexHull(contour_points, hull_points);

  // Compute a bounded rectangle around the convex hull.
  const cv::Rect bounded_rect = cv::boundingRect(hull_points);

  // Crop bounded rectangle from input_frame (minimizes computational cost 
  // later on since we're searching over a smaller area).
  const cv::Mat input_frame_mat = formats::MatView(&input_frame);
  const cv::Mat cropped_input_region = input_frame_mat(bounded_rect);

  // Translate hull points into bounded rect frame 
  // (previously in input image frame).
  const mobio::Point translation_offset(-bounded_rect.x, -bounded_rect.y);
  RET_CHECK_OK(TranslatePoints(&hull_points, translation_offset)) << "Error encountered while translating hull points.";;

  // Create a test matrix of size cropped_input_region and fill it with 
  // value of kColourOutsideContour. 
  cv::Mat contour_mask_mat = cv::Mat::ones(cropped_input_region.size(), 
                                            CV_8U)*kColourOutsideContour;
  
  
  // Apply fillConvexPoly to colour all pixels within contour in test matrix.
  cv::fillConvexPoly(contour_mask_mat, hull_points, 
                      cv::Scalar(kColourInsideContour));
  
  // Calculate mean intensity of pixels within contour.
  auto mean_intensity_ptr = absl::make_unique<mobio::Intensity>();
  ComputeMeanIntensityContour(cropped_input_region, contour_mask_mat, mean_intensity_ptr);

  // Publish outputs.
  cc->Outputs().Tag(kIntensityTag).Add(mean_intensity_ptr.release(), cc->InputTimestamp());
  LOG(INFO) << "Published intensity output";


  return absl::OkStatus();
}

} // namespace mediapipe
