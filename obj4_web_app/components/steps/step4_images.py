# obj4_web_app/components/steps/step4_images.py
"""
Step 4: Generate Images
"""

import streamlit as st
from pathlib import Path
from ..state_manager import get_step_data, update_step_data, go_to_next_step, go_to_previous_step


def render_step4_content(design_generator, reference_images_dir: Path, clip_threshold: float = 0.80) -> None:
    """
    Render Step 4: Generate Images form.

    Args:
        design_generator: DesignGeneratorWrapper instance
        reference_images_dir: Path to reference images directory
        clip_threshold: CLIP similarity threshold
    """
    data = get_step_data(4)
    step3_data = get_step_data(3)
    step1_data = get_step_data(1)

    prompt = step3_data.get('generated_prompt', '')

    if not prompt:
        st.error("No prompt available. Please complete Step 3 first.")
        return

    # Reference image selection
    st.markdown("**Reference Image:**")

    available_refs = list(reference_images_dir.glob("lulu_pig_ref_*.png")) + \
                     list(reference_images_dir.glob("lulu_pig_ref_*.jpg"))

    if not available_refs:
        st.warning("No reference images found in data/reference_images/")
        return

    ref_names = [ref.name for ref in available_refs]
    current_ref = data.get('reference_image', ref_names[0] if ref_names else '')

    selected_ref = st.selectbox(
        "Select Reference",
        options=ref_names,
        index=ref_names.index(current_ref) if current_ref in ref_names else 0,
        key="step4_ref_select",
        label_visibility="collapsed"
    )

    update_step_data(4, {'reference_image': selected_ref})

    # Show reference image
    with st.expander("View Reference Image"):
        ref_path = reference_images_dir / selected_ref
        st.image(str(ref_path), width=200)

    st.markdown("---")

    # Generation settings
    col1, col2 = st.columns(2)

    with col1:
        num_images = st.slider(
            "Number of Images",
            min_value=1, max_value=4,
            value=data.get('num_images', 2),
            key="step4_num_images"
        )
        update_step_data(4, {'num_images': num_images})

    with col2:
        mode_options = {
            "single": "Micro Variations",
            "preset": "Theme Scenes",
            "creative": "AI Creative"
        }
        variation_mode = st.selectbox(
            "Variation Mode",
            options=list(mode_options.keys()),
            format_func=lambda x: mode_options[x],
            index=list(mode_options.keys()).index(data.get('variation_mode', 'single')),
            key="step4_mode"
        )
        update_step_data(4, {'variation_mode': variation_mode})

    st.markdown("---")

    # Generate button
    if design_generator is None:
        st.warning("Design Generator not available. Check API configuration.")
    else:
        if st.button(f"Generate {num_images} Images", key="step4_generate",
                    type="primary", use_container_width=True):
            _generate_images(
                design_generator=design_generator,
                prompt=prompt,
                ref_path=reference_images_dir / selected_ref,
                num_images=num_images,
                variation_mode=variation_mode,
                character_name=step1_data.get('character_name', ''),
                character_desc=step1_data.get('character_desc', ''),
                clip_threshold=clip_threshold
            )

    # Show generated images
    images = data.get('generated_images', [])
    if images:
        st.markdown("---")
        st.markdown("**Generated Images:**")

        successful = [img for img in images if img.get('success')]

        if successful:
            cols = st.columns(2)
            for i, img in enumerate(successful):
                with cols[i % 2]:
                    st.image(img['image'], use_container_width=True)
                    clip = img.get('clip_similarity', 0)
                    color = "green" if clip >= clip_threshold else "orange"
                    st.markdown(f"CLIP: :{color}[{clip:.4f}]")

                    # Download button
                    if design_generator:
                        img_bytes = design_generator.image_to_bytes(img['image'])
                        st.download_button(
                            "Download",
                            data=img_bytes,
                            file_name=f"design_{i+1}.png",
                            mime="image/png",
                            key=f"step4_download_{i}"
                        )

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("<- Back", key="step4_back", use_container_width=True):
            go_to_previous_step()
            st.rerun()

    with col3:
        successful_images = [img for img in data.get('generated_images', []) if img.get('success')]
        is_valid = len(successful_images) > 0
        if st.button("Next: 完成 ->", key="step4_next", type="primary",
                    use_container_width=True, disabled=not is_valid):
            go_to_next_step()
            st.rerun()

    if not is_valid:
        st.warning("Please generate at least one image.")


def _generate_images(design_generator, prompt, ref_path, num_images, variation_mode,
                     character_name, character_desc, clip_threshold):
    """Generate images and update state."""
    from obj4_web_app.utils.design_generator import DesignGenerationError

    progress = st.progress(0)
    status = st.empty()

    def update_progress(p, msg):
        progress.progress(p)
        status.text(msg)

    try:
        results = design_generator.generate_designs(
            prompt=prompt,
            reference_image_path=str(ref_path),
            num_images=num_images,
            progress_callback=update_progress,
            max_retries=3,
            use_multithreading=True,
            variation_mode=variation_mode,
            character_name=character_name,
            character_desc=character_desc
        )

        update_step_data(4, {'generated_images': results})

        progress.empty()
        status.empty()

        successful = sum(1 for r in results if r.get('success'))
        if successful == num_images:
            st.success(f"Generated {successful}/{num_images} images!")
        elif successful > 0:
            st.warning(f"Generated {successful}/{num_images} images")
        else:
            st.error("All generation failed")

        st.rerun()

    except DesignGenerationError as e:
        progress.empty()
        status.empty()
        st.error(f"Error: {str(e)}")
