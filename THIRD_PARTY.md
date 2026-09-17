# Model and test-asset provenance

- YOLO11 pretrained Detect/Pose and BoT-SORT integration: [Ultralytics](https://github.com/ultralytics/ultralytics), [tracking docs](https://docs.ultralytics.com/modes/track/). Weights `yolo11n.pt` and `yolo11n-pose.pt` downloaded through Ultralytics from its official assets release. Consult upstream AGPL-3.0/commercial licensing for distribution.
- Hand-object network: Shan, Geng, Shu, Fouhey, **Understanding Human Hands in Contact at Internet Scale**, CVPR 2020. [Original source](https://github.com/ddshan/hand_object_detector), [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.html), [dataset/project](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/).
- Pure-PyTorch implementation: [larsrpe/hand_object_detector_pytorch](https://github.com/larsrpe/hand_object_detector_pytorch), pinned commit `ff6303e6e99acd710fcc36e9fba0da606cea3551`. Original MIT license is retained in `external/hand_object_detector_pytorch/LICENSE`. Project adapter inference follows the architecture's box decoding and published hand-object matching rule.
- Contact weights: `faster_rcnn_1_8_132028.pth`, 378,147,662 bytes, 100DOH+ego model. Both original Google Drive links failed during setup, consistent with [upstream issue 38](https://github.com/ddshan/hand_object_detector/issues/38). Retrieved from [highbyte/frankmocap's archived revision](https://huggingface.co/highbyte/frankmocap/tree/1f9f0d3c77e74ebf1b86ad7362cebda09d82ffb3). SHA-256 `3444e3b992417cb6adfcd71539ca8c92602c21a3e2fdfff4628dbdbdf8a2dd03` matches that mirror's Git-LFS hash, not an author-signed checksum. The mirror metadata lists CC-BY-NC-2.0; consult the original model/dataset terms as well. The upstream authors request citations/conditions for Epic-Kitchens, EGTEA and CharadesEgo when using the ego-trained model.
- Real-model smoke-test still: [Ultralytics bus image](https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg). The local MP4 repeats this image solely to test video integration; it is not a recorded backpack event.
- Contact smoke-test image: `assets/boardgame_848_sU8S98MT1Mo_00013957.png` in the pinned contact repository. This is an already annotated upstream example; its use here tests executable inference, not quantitative accuracy.
- Demo input: generated locally by `utils/demo.py`. Its observations are scripted and clearly identified as synthetic throughout the output.
# Person appearance ReID

Added at the user's request to distinguish the pickup person from a different
returning person. Intel Open Model Zoo `person-reidentification-retail-0288`,
OmniScaleNet-based pretrained person encoder, executed with OpenVINO on CPU.

- Documentation: https://docs.openvino.ai/2023.3/omz_models_model_person_reidentification_retail_0288.html
- Official weight manifest and SHA-384 checksums: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0288/model.yml
- Weight release: official Open Model Zoo 2023.0 FP32 files from storage.openvinotoolkit.org.
- License: Apache-2.0, https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE
- Exact download URLs and hashes are pinned in `scripts/setup_reid.py`.

No custom training, face recognition, persistent identity store or downloaded
third-party executable Python source is used for this component.
