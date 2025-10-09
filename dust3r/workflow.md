# AsymmetricCroCo3DStereo Model Data Flow

## 1. Initialization
- **`__init__`**
  - Sets up model configurations
  - Calls `set_downstream_head` to initialize downstream heads
  - Calls `set_freeze` to determine which parameters to freeze

## 2. Forward Pass
- **`forward(view1, view2)`**
  - ### 2.1 Encoding Symmetrized Images
    - **`_encode_symmetrized(view1, view2)`**
      - #### 2.1.1 Encoding Image Pairs
        - **`_encode_image_pairs(img1, img2, true_shape1, true_shape2)`**
          - ##### 2.1.1.1 Encoding Single Image
            - **`_encode_image(image, true_shape)`**
              - `self.patch_embed(image, true_shape)`
              - `self.enc_blocks`
              - `self.enc_norm(x)`
            - Returns encoded features (`x`) and positional embeddings (`pos`)
          - Concatenates and processes images if they have the same shape; otherwise, processes separately
        - Returns encoded features and positional embeddings for both images
      - If views are symmetrized, interleaves features and positional embeddings
    - Returns shapes, features, and positional embeddings for both views

  - ### 2.2 Decoding
    - **`_decoder(feat1, pos1, feat2, pos2)`**
      - Projects features to decoder dimensions
      - Iterates through decoder blocks (`self.dec_blocks` and `self.dec_blocks2`)
      - Normalizes final output
    - Returns decoded outputs for both images

  - ### 2.3 Downstream Heads
    - **`_downstream_head(head_num, decout, img_shape)`**
      - Uses downstream head (`self.head1` or `self.head2`) to process decoded output
    - Returns results for both images

  - Combines results, adjusts second result to be in the first view's frame
  - Returns final results (`res1` and `res2`)
