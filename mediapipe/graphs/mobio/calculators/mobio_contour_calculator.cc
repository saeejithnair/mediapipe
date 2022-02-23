#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

#include "mediapipe/graphs/mobio/constants.h"

namespace mediapipe {

namespace {

// Input stream packet tag names.
constexpr char kImageTag[] = "IMAGE";
constexpr char kLandmarksTag[] = "LANDMARKS";
// Input side packet tag names.
constexpr char kNumFacesTag[] = "NUM_FACES";
// Output stream packet tag names.
constexpr char kContourIntensityForeheadTag[] = "CONTOUR_INTENSITY_FOREHEAD";
constexpr char kContourIntensityLCheekTag[] = "CONTOUR_INTENSITY_LCHEEK";
constexpr char kContourIntensityRCheekTag[] = "CONTOUR_INTENSITY_RCHEEK";

// Indices used for tracking landmark points in tri-region contours.
// These points were empirically selected to maximize coverage of each region
// while minimizing capture of non-smooth, non-skin artifacts 
// (eyebrows, facial hair, hairline, nose bridge, etc.)
constexpr std::array kTrackerIdxsIntensityForehead {103, 67,109, 10, 338, 297, 332, 334, 296, 336, 107, 66, 105, 104, 69, 108, 151, 337, 299, 333};
constexpr std::array kTrackerIdxsIntensityRightCheek {116, 111, 117, 118, 101, 203, 206, 216, 207, 205, 187, 147, 123, 50};
constexpr std::array kTrackerIdxsIntensityLeftCheek {345, 340, 346, 347, 330, 266, 423, 426, 436, 427, 411, 376, 352, 280, 425};

typedef std::vector<NormalizedLandmarkList> NormalizedLandmarkLists;
} // anonymous namespace

class MobioContourCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  // Computes contour points in image frame (i.e. pixels) from
  // NormalizedLandmark points (i.e. [0,1)). Stores computed
  // contour points in output vector (contour_points).
  absl::Status ComputeContourFromTrackers(
                const int img_width, const int img_height,
                const NormalizedLandmarkList& landmark_list, 
                const std::vector<int>& trackers,
                std::unique_ptr<mobio::Points>& contour_points) const;

  // Returns absl::OkStatus if pixel is within bounds of image.
  absl::Status CheckPixelWithinImage(const int px, 
                                     const int py, 
                                     const int img_width, 
                                     const int img_height) const;

  // Populates hashmap mapping output tag names to tracker arrays.
  // This function must be updated every time a new output stream is added.
  absl::Status PopulateOutputsToTrackersMap(void);

  // Updates the timestamp bound for all registered output tag names.
  // Always returns absl::StatusOk(). 
  absl::Status UpdateTimestampBoundAndReturnOk(CalculatorContext* cc);

  // Map registering output tags to trackers. Also used to keep track of
  // which output streams have been registered.
  std::unordered_map<std::string, std::vector<int>> outputs_to_trackers_map_;
};

REGISTER_CALCULATOR(MobioContourCalculator);

absl::Status MobioContourCalculator::GetContract(CalculatorContract* cc) {
  // Validate input/output stream tags connected to node.
  RET_CHECK(cc->Inputs().HasTag(kImageTag));
  RET_CHECK(cc->Inputs().HasTag(kLandmarksTag));
  RET_CHECK(cc->InputSidePackets().HasTag(kNumFacesTag));
  RET_CHECK(cc->Outputs().HasTag(kContourIntensityForeheadTag));
  RET_CHECK(cc->Outputs().HasTag(kContourIntensityLCheekTag));
  RET_CHECK(cc->Outputs().HasTag(kContourIntensityRCheekTag));
  
  // Set type for input stream packets.
  cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkLists>();

  // Set type for num faces side packet.
  cc->InputSidePackets().Tag(kNumFacesTag).Set<int>();

  // Set type for output stream packets.
  cc->Outputs().Tag(kContourIntensityForeheadTag).Set<mobio::Points>();
  cc->Outputs().Tag(kContourIntensityLCheekTag).Set<mobio::Points>();
  cc->Outputs().Tag(kContourIntensityRCheekTag).Set<mobio::Points>();

  return absl::OkStatus();
}

absl::Status MobioContourCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  RET_CHECK_OK(PopulateOutputsToTrackersMap());

  return absl::OkStatus();
}

absl::Status MobioContourCalculator::PopulateOutputsToTrackersMap() {
  outputs_to_trackers_map_.insert({kContourIntensityForeheadTag, std::vector<int>(
                                      kTrackerIdxsIntensityForehead.begin(), 
                                      kTrackerIdxsIntensityForehead.end())});
  outputs_to_trackers_map_.insert({kContourIntensityLCheekTag, std::vector<int>(
                                      kTrackerIdxsIntensityLeftCheek.begin(), 
                                      kTrackerIdxsIntensityLeftCheek.end())});
  outputs_to_trackers_map_.insert({kContourIntensityRCheekTag, std::vector<int>(
                                      kTrackerIdxsIntensityRightCheek.begin(), 
                                      kTrackerIdxsIntensityRightCheek.end())});

  return absl::OkStatus();
}

// Validate that pixels are within bounds of image frame.
absl::Status MobioContourCalculator::CheckPixelWithinImage(
                const int px, const int py, 
                const int img_width, const int img_height) const {
  // Pixel x-coordinate must be smaller than image width.
  RET_CHECK_LT(px, img_width) << "Failed bounds check. Expected pixel (" << px 
                              << ") < image width (" << img_width << ").";
  // Pixel y-coordinate must be smaller than image hegiht.
  RET_CHECK_LT(py, img_height) << "Failed bounds check. Expected pixel (" << py 
                               << ") < image height (" << img_height << ").";
  // Pixel x-coordinate must be greater than or equal to 0.
  RET_CHECK_GE(px, 0) << "Failed bounds check. Expected pixel (" << px << ") >= (0).";
  // Pixel y-coordinate must be greater than or equal to 0.
  RET_CHECK_GE(py, 0) << "Failed bounds check. Expected pixel (" << py << ") >= (0).";

  return absl::OkStatus();
}

absl::Status MobioContourCalculator::ComputeContourFromTrackers(
    const int img_width, const int img_height,
    const NormalizedLandmarkList& landmark_list, 
    const std::vector<int>& trackers,
    std::unique_ptr<mobio::Points>& contour_points) const {
  contour_points->reserve(trackers.size());
  
  for (int i=0; i<trackers.size(); ++i) {
    const auto& landmark = landmark_list.landmark(trackers[i]);
    int px = int(landmark.x()*img_width);
    int py = int(landmark.y()*img_height);

    RET_CHECK_OK(CheckPixelWithinImage(px, py, img_width, img_height));

    contour_points->emplace_back(mobio::Point(px, py));
  }

  return absl::OkStatus();
}

absl::Status MobioContourCalculator::UpdateTimestampBoundAndReturnOk(CalculatorContext* cc) {
  for (const auto &[output_tag_name, trackers]: outputs_to_trackers_map_ ) {
    cc->Outputs().Tag(output_tag_name).SetNextTimestampBound(
                    cc->InputTimestamp().NextAllowedInStream());
  }

  return absl::OkStatus();
}

absl::Status MobioContourCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageTag).IsEmpty() || 
      cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
    LOG(WARNING) << "Missing inputs.";

    return UpdateTimestampBoundAndReturnOk(cc);
  }

  const auto& input_frame = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
  const auto input_frame_width = input_frame.Width();
  const auto input_frame_height = input_frame.Height();

  // Get vector of landmark list.
  const auto& landmark_lists = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkLists>();

  // Log warning if more than 1 face is detected. 
  const auto num_faces = cc->InputSidePackets().Tag(kNumFacesTag).Get<int>();
  if (num_faces != 1) {
    LOG(WARNING) << "Expected only 1 face, received " << num_faces << " faces.";
    return UpdateTimestampBoundAndReturnOk(cc);
  }

  // Get face landmarks.
  const auto& landmark_list = landmark_lists.front();

  for (const auto &[output_tag_name, trackers]: outputs_to_trackers_map_ ) {
    auto contour_points_ptr = absl::make_unique<mobio::Points>();
    auto status = ComputeContourFromTrackers(input_frame_width,
                                             input_frame_height,
                                             landmark_list, 
                                             trackers,
                                             contour_points_ptr);

    if (!status.ok()) {
      LOG(WARNING) << "Failed to compute contour for " << output_tag_name 
                   << ". " << status;

      // Log error as warning and returns Ok. Contour computation usually 
      // fails when subject has part of their face outside camera view.
      // Instead of crashing the app, this will just skip processing for
      // the current frame.
      return UpdateTimestampBoundAndReturnOk(cc);
    }

    // Publish to output node.
    cc->Outputs().Tag(output_tag_name).Add(
          contour_points_ptr.release(), cc->InputTimestamp());
  }
  
  return absl::OkStatus();
}

} // namespace mediapipe
