#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"

#include "mediapipe/graphs/mobio/constants.h"

namespace mediapipe {

namespace mobio {
namespace {

constexpr char kImageTag[] = "IMAGE";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kNormalizedRectanglesTag[] = "MOBIO_NORM_RECTS";
constexpr char kDetectionsTag[] = "MOBIO_DETECTIONS";
typedef std::vector<NormalizedLandmarkList> NormalizedLandmarkLists;
} // anonymous namespace

class MobioResultProcessorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:

};

REGISTER_CALCULATOR(MobioResultProcessorCalculator);

absl::Status MobioResultProcessorCalculator::GetContract(CalculatorContract* cc) {

  RET_CHECK(cc->Inputs().HasTag(kImageTag));
  RET_CHECK(cc->Outputs().HasTag(kImageTag));
  RET_CHECK(cc->Inputs().HasTag(kLandmarksTag));
  
  
  cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkLists>();
  cc->Outputs().Tag(kImageTag).Set<ImageFrame>();
  // RET_CHECK(cc->Inputs().HasTag(kNormalizedRectanglesTag));
  // RET_CHECK(cc->Inputs().HasTag(kDetectionsTag));
  // RET_CHECK(cc->Outputs().HasTag(kImageTag));

  // cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();

  return absl::OkStatus();
}

absl::Status MobioResultProcessorCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));


  return absl::OkStatus();
}

absl::Status MobioResultProcessorCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_frame = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();

  // Get vector of landmark list.
  const auto& landmarks_lists = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkLists>();

  // Pop first element to get vector of landmarks corresponding to a face.
  // Can be extended to handle multiple faces, 
  // but for demo purposes we only care about the first face.
  const auto& landmarks_list = landmarks_lists.front();
  LOG(INFO) << "Got landmarks lists of size " << landmarks_list.landmark_size();
  for (int i=0; i<landmarks_list.landmark_size(); ++i) {
    const auto& landmark = landmarks_list.landmark(i);
    LOG(INFO) << "x: " << landmark.x() << ", y: " << landmark.y() << "z: " << landmark.z();
  }

  RET_CHECK(0);

  return absl::OkStatus();
}

} // namespace mobio
} // namespace mediapipe
