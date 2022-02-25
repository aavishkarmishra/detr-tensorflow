# DETR : End-to-End Object Detection with Transformers (Tensorflow)
DETR, which stands for Detection Transformers, was proposed by a team from the Facebook AI group, and it is, as of today, a radical shift from the current approaches to perform Deep Learning based Object Detection.

Instead of filtering and refining a set of object proposals, as done by two-stage techniques like Faster-RCNN and its adaptations, or generating dense detection grids, as done by single-stage techniques like SSD and YOLO, DETR frames the detection problem as an image to set mapping. With this formulation, both the architecture and the training process become significantly simpler. There is no need for hand-designed anchor matching schemes or post-processing steps like Non Max Suppression to discard redundant detections.

DETR uses a CNN backbone to extract a higher level feature representation of the image, which is then fed into a Transformer model. The Transformer Encoder is responsible for processing this image representation, while the Decoder maps a fixed set of learned object queries to detections, performing attention over the Encoder's output.

DETR is trained with a set-based global loss that finds a bipartite matching between the set of detections and ground-truth objects (non matched detections are assigned to a special no object class), which in turn forces unique detections.
