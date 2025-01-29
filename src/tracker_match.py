import torch

def calculate_center(bbox):
    """Calculate the center of a bounding box."""
    x1, y1, x2, y2 = bbox[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def reorder_tracking_boxes(original_boxes, tracking_boxes, ids):
    """Reorder tracking boxes to match the order of the original boxes."""
    reordered_tracking_boxes = []
    updated_indices = []
    used_indices = set()

    for tracking_bbox in tracking_boxes:
        tracking_center = calculate_center(tracking_bbox)
        closest_box = None
        closest_distance = float('inf')
        closest_index = 1

        for original_idx, original_bbox in enumerate(original_boxes):
            if original_idx in used_indices:
                continue

            original_center = calculate_center(original_bbox)
            distance = ((tracking_center[0] - original_center[0]) ** 2 +
                        (tracking_center[1] - original_center[1]) ** 2) ** 0.5

            if distance < closest_distance:
                closest_distance = distance
                closest_box = original_bbox
                closest_index = original_idx

        reordered_tracking_boxes.append(closest_box.tolist())
        updated_indices.append(ids[closest_index])
        used_indices.add(closest_index)

        reordered_tracking_boxes = list(filter(lambda x: x is not None, reordered_tracking_boxes))
    return torch.Tensor(reordered_tracking_boxes), updated_indices

if __name__ == '__main__':
    # Example usage
    original_boxes = [
        [667, 228, 765, 362, 0.8, 0, 0],
        [660, 221, 733, 370, 0.9, 0, 0],
        [600, 200, 700, 350, 0.85, 0, 0],
        [650, 250, 750, 400, 0.95, 0, 0]
    ]
    tracking_boxes = [
        [661, 222, 734, 371, 0.85, 0, 0],
        [666, 229, 764, 361, 0.8, 0, 0],
        [601, 201, 701, 351, 0.86, 0, 0],
        [651, 251, 751, 401, 0.96, 0, 0]
    ]
    idx = [1, 2, 3, 4]
    reordered_boxes, updated_indices = reorder_tracking_boxes(original_boxes, tracking_boxes, idx)
    print(reordered_boxes)
    print(updated_indices)

