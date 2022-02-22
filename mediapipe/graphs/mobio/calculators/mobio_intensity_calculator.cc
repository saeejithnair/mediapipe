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
constexpr char kContourTag[] = "CONTOUR";
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
                               const cv::Point2d& translation_offset);

  // Calculates the mean intensity of pixels within contour region inside image.
  // Pixel is inside contour if mask[i,j] has non-zero value.
  mobio::Intensity ComputeMeanIntensityContour(
                      const cv::Mat& image,
                      const cv::Mat& mask);

};

REGISTER_CALCULATOR(MobioIntensityCalculator);

absl::Status MobioIntensityCalculator::GetContract(CalculatorContract* cc) {
  // Validate input/output stream tags connected to node.
  RET_CHECK(cc->Inputs().HasTag(kImageTag));
  RET_CHECK(cc->Inputs().HasTag(kContourTag));
  RET_CHECK(cc->Outputs().HasTag(kIntensityTag));
  
  // Set type for input stream packets.
  cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Inputs().Tag(kContourTag).Set<mobio::Points>();

  // Set type for output stream packets.
  cc->Outputs().Tag(kIntensityTag).Set<mobio::Intensity>();

  return absl::OkStatus();
}

absl::Status MobioIntensityCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}

absl::Status MobioIntensityCalculator::TranslatePoints(
                mobio::Points* points, const cv::Point2d& translation_offset) {
  RET_CHECK(points != nullptr);

  for (auto& point: *points) {
    point += translation_offset;
  }

  return absl::OkStatus();
}

mobio::Intensity MobioIntensityCalculator::ComputeMeanIntensityContour(
                const cv::Mat& image,
                const cv::Mat& mask) {
  const cv::Scalar mean_intensity = cv::mean(image, mask);

  return mobio::Intensity{mean_intensity[0], mean_intensity[1], mean_intensity[2]};
}

absl::Status MobioIntensityCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().Tag(kContourTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_frame = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
  const auto input_frame_width = input_frame.Width();
  const auto input_frame_height = input_frame.Height();

  // Get vector of points enclosing contour region.
  const auto& contour_points = cc->Inputs().Tag(kContourTag).Get<mobio::Points>();

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
  const cv::Point2d translation_offset(-bounded_rect.x, -bounded_rect.y);
  RET_CHECK_OK(TranslatePoints(&hull_points, translation_offset));

  // Create a test matrix of size cropped_input_region and fill it with 
  // value of kColourOutsideContour. 
  cv::Mat contour_mask_mat = cv::Mat::ones(cropped_input_region.size(), 
                                            CV_8U)*kColourOutsideContour;
  
  
  // Apply fillConvexPoly to colour all pixels within contour in test matrix.
  cv::fillConvexPoly(contour_mask_mat, hull_points, 
                      cv::Scalar(kColourInsideContour));
  
  // Calculate mean intensity of pixels within contour.
  const auto mean_intensity = ComputeMeanIntensityContour(cropped_input_region, contour_mask_mat);

  // Publish outputs.
  cc->Outputs().Tag(kIntensityTag).Add(&mean_intensity, cc->InputTimestamp());

  return absl::OkStatus();
}

} // namespace mediapipe
