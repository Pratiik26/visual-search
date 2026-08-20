/**
 * AURA DIAMONDS — Haute Joaillerie Visual Search Studio Client
 * Nyris Multi-Object Hotspots & Interactive Target Crop Frame
 */

document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('upload-dropzone');
  const fileInput = document.getElementById('ring-file-input');
  const dropzoneIdle = document.getElementById('dropzone-idle');
  const dropzonePreview = document.getElementById('dropzone-preview');
  const previewViewport = document.getElementById('preview-viewport');
  const previewImg = document.getElementById('ring-preview-img');
  const removeImgBtn = document.getElementById('remove-img-btn');
  const triggerSearchBtn = document.getElementById('trigger-search-btn');
  const clearSearchBtn = document.getElementById('clear-search-btn');


  // Interactive Target Frame & Hotspots
  const hotspotPinsLayer = document.getElementById('hotspot-pins-layer');
  const targetCropFrame = document.getElementById('target-crop-frame');
  const cropBadge = document.getElementById('crop-badge');
  const targetFramePanel = document.getElementById('target-frame-panel');
  const hotspotChipsRow = document.getElementById('hotspot-chips-row');
  const applyFrameSearchBtn = document.getElementById('apply-frame-search-btn');
  const resetFrameBtn = document.getElementById('reset-frame-btn');

  // Grown Brilliance Interactive HUD Chips
  const recognitionHud = document.getElementById('recognition-hud');
  const hudShapeSelect = document.getElementById('hud-shape-select');
  const hudMetalSelect = document.getElementById('hud-metal-select');
  const hudArchSelect = document.getElementById('hud-arch-select');
  const hudProngSelect = document.getElementById('hud-prong-select');
  const hudStyleSelect = document.getElementById('hud-style-select');

  // Display Areas
  const searchLoading = document.getElementById('search-loading');
  const searchEmpty = document.getElementById('search-empty');
  const resultsGrid = document.getElementById('results-cards-grid');

  const tabCardsBtn = document.getElementById('tab-cards-btn');
  const tabRawBtn = document.getElementById('tab-raw-btn');
  const jsonViewer = document.getElementById('json-viewer');
  const jsonOutputText = document.getElementById('json-output-text');
  const copyJsonAction = document.getElementById('copy-json-action');

  const videoModal = document.getElementById('video-modal-backdrop');
  const modalVideo = document.getElementById('modal-video');
  const modalTitle = document.getElementById('video-modal-title');
  const modalClose = document.getElementById('video-modal-close');

  let activeFile = null;
  let activeUrl = null;
  let lastResultData = null;
  let isUpdatingHud = false;
  let detectedRegionsList = [];
  let searchDebounceTimer = null;

  // Active Crop Frame State (Normalized 0.0 to 1.0)
  let activeCropBox = null;
  let isDraggingFrame = false;
  let isResizingFrame = false;
  let activeHandle = null;
  let dragStartX = 0, dragStartY = 0;
  let initialFrameRect = null;

  // File Upload Handlers
  dropzoneIdle.addEventListener('click', () => {
    fileInput.click();
  });

  dropzone.addEventListener('click', (e) => {
    if (dropzoneIdle.style.display !== 'none' && e.target !== removeImgBtn) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  });

  removeImgBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearQueryImage();
  });

  clearSearchBtn.addEventListener('click', () => {
    clearQueryImage();
    showEmpty();
  });

  function processFile(file) {
    if (!file.type.match('image.*')) {
      alert('Please upload an image file (JPG, PNG, WebP).');
      return;
    }
    activeFile = file;
    activeUrl = null;
    activeCropBox = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropzoneIdle.style.display = 'none';
      dropzonePreview.style.display = 'flex';
      triggerSearchBtn.disabled = false;
      previewImg.onload = () => {
        executeSearch(true);
      };
    };
    reader.readAsDataURL(file);
  }

  function clearQueryImage() {
    activeFile = null;
    activeUrl = null;
    activeCropBox = null;
    detectedRegionsList = [];
    fileInput.value = '';
    previewImg.src = '';
    if (hotspotPinsLayer) hotspotPinsLayer.innerHTML = '';
    if (hotspotChipsRow) hotspotChipsRow.innerHTML = '';
    if (targetCropFrame) targetCropFrame.style.display = 'none';
    if (targetFramePanel) targetFramePanel.style.display = 'none';
    dropzoneIdle.style.display = 'block';
    dropzonePreview.style.display = 'none';
    triggerSearchBtn.disabled = true;
    recognitionHud.style.display = 'none';
  }

  // Quick Samples
  document.querySelectorAll('.sample-card').forEach(btn => {
    btn.addEventListener('click', () => {
      const url = btn.getAttribute('data-url');
      activeUrl = url;
      activeFile = null;
      activeCropBox = null;
      previewImg.src = url;
      dropzoneIdle.style.display = 'none';
      dropzonePreview.style.display = 'flex';
      triggerSearchBtn.disabled = false;
      previewImg.onload = () => {
        executeSearch(true);
      };
    });
  });

  triggerSearchBtn.addEventListener('click', () => {
    executeSearch(false);
  });

  applyFrameSearchBtn.addEventListener('click', () => {
    executeSearch(true);
  });

  resetFrameBtn.addEventListener('click', () => {
    activeCropBox = null;
    if (targetCropFrame) targetCropFrame.style.display = 'none';
    executeSearch(true);
  });

  function debouncedSearch(isNewQuery = false, delayMs = 90) {
    if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      executeSearch(isNewQuery);
    }, delayMs);
  }

  // Interactive Chip Change Handlers (Debounced for fast response)
  [hudShapeSelect, hudMetalSelect, hudArchSelect, hudProngSelect, hudStyleSelect].forEach(chip => {
    chip.addEventListener('change', () => {
      if (!isUpdatingHud && (activeFile || activeUrl)) {
        debouncedSearch(false, 60);
      }
    });
  });

  // ==========================================================================
  // CORE SEARCH EXECUTION
  // ==========================================================================
  async function executeSearch(isNewQuery = false) {
    if (!activeFile && !activeUrl) return;

    showLoading();

    try {
      let resp;
      const shapeVal = isNewQuery ? null : hudShapeSelect.value;
      const metalVal = isNewQuery ? null : hudMetalSelect.value;
      const archVal = isNewQuery ? null : hudArchSelect.value;
      const prongVal = isNewQuery ? null : hudProngSelect.value;
      const styleVal = isNewQuery ? null : hudStyleSelect.value;

      if (activeFile) {
        const fd = new FormData();
        fd.append('file', activeFile);
        fd.append('top_k', 12);
        if (shapeVal) fd.append('shape', shapeVal);
        if (metalVal) fd.append('band_color', metalVal);
        if (archVal) fd.append('band_architecture', archVal);
        if (prongVal) fd.append('prong_style', prongVal);
        if (styleVal) fd.append('style', styleVal);

        if (activeCropBox) {
          fd.append('crop_x1', activeCropBox.x1);
          fd.append('crop_y1', activeCropBox.y1);
          fd.append('crop_x2', activeCropBox.x2);
          fd.append('crop_y2', activeCropBox.y2);
        }

        resp = await fetch('/aiSearchProducts', {
          method: 'POST',
          body: fd
        });
      } else {
        const payload = {
          image_url: activeUrl,
          top_k: 12,
          shape: shapeVal || null,
          band_color: metalVal || null,
          band_architecture: archVal || null,
          prong_style: prongVal || null,
          style: styleVal || null
        };

        if (activeCropBox) {
          payload.crop_x1 = activeCropBox.x1;
          payload.crop_y1 = activeCropBox.y1;
          payload.crop_x2 = activeCropBox.x2;
          payload.crop_y2 = activeCropBox.y2;
        }

        resp = await fetch('/api/v1/search/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      }

      if (!resp.ok) {
        throw new Error(`Search failed: HTTP ${resp.status}`);
      }

      const data = await resp.json();
      lastResultData = data;

      // Update AI HUD Filters
      if (data.query_analysis || data.uploaded_image_metadata) {
        updateHud(data.query_analysis || {}, data.uploaded_image_metadata || data.query_metadata || {});
      }

      // Render Detected Object Hotspot Pins & Frame
      if (data.region_detection) {
        renderObjectHotspots(data.region_detection);
      }

      // Render Results
      renderResults(data.results || [], data.query_analysis);

      // JSON Raw Tab
      jsonOutputText.textContent = JSON.stringify(data, null, 2);

    } catch (err) {
      console.error(err);
      resultsGrid.innerHTML = `
        <div class="error-box" style="grid-column: 1/-1; padding: 2rem; background: #FFF5F5; border: 1px solid #FED7D7; border-radius: 8px;">
          <p>⚠️ Error performing visual search: ${err.message}</p>
        </div>
      `;
      searchLoading.style.display = 'none';
      resultsGrid.style.display = 'grid';
    }
  }

  // ==========================================================================
  // OBJECT AUTO-DETECTION & INTERACTIVE BOUNDING BOX
  // ==========================================================================
  function renderObjectHotspots(regionData) {
    if (hotspotPinsLayer) hotspotPinsLayer.innerHTML = '';
    if (hotspotChipsRow) hotspotChipsRow.innerHTML = '';

    const regions = regionData.all_regions || [regionData.detected_region];
    detectedRegionsList = regions;
    targetFramePanel.style.display = 'flex';

    // Auto-fit crop frame to primary detected ring region
    if (!activeCropBox && regions.length > 0) {
      const primary = regions[0].region || regions[0];
      activeCropBox = {
        x1: primary.rel_left,
        y1: primary.rel_top,
        x2: primary.rel_right,
        y2: primary.rel_bottom
      };
      setCropFrameUI(primary.rel_left, primary.rel_top, primary.rel_right, primary.rel_bottom, 'Diamond Ring');
    } else if (activeCropBox) {
      setCropFrameUI(activeCropBox.x1, activeCropBox.y1, activeCropBox.x2, activeCropBox.y2, 'Custom Target Frame');
    }
  }

  function setCropFrameUI(x1, y1, x2, y2, label = 'Target Object') {
    const imgW = previewImg.clientWidth || previewViewport.offsetWidth || 380;
    const imgH = previewImg.clientHeight || previewViewport.offsetHeight || 380;

    const leftPx = Math.max(0, x1 * imgW);
    const topPx = Math.max(0, y1 * imgH);
    const widthPx = Math.min(imgW - leftPx, (x2 - x1) * imgW);
    const heightPx = Math.min(imgH - topPx, (y2 - y1) * imgH);

    targetCropFrame.style.left = `${leftPx}px`;
    targetCropFrame.style.top = `${topPx}px`;
    targetCropFrame.style.width = `${widthPx}px`;
    targetCropFrame.style.height = `${heightPx}px`;
    targetCropFrame.style.display = 'block';

    cropBadge.textContent = `🎯 ${label}`;
  }

  // ==========================================================================
  // DRAG & RESIZE BOUNDING BOX INTERACTION
  // ==========================================================================
  targetCropFrame.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('crop-handle') || e.target.classList.contains('crop-edge')) {
      isResizingFrame = true;
      activeHandle = e.target.getAttribute('data-handle');
    } else {
      isDraggingFrame = true;
    }
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    initialFrameRect = {
      left: targetCropFrame.offsetLeft,
      top: targetCropFrame.offsetTop,
      width: targetCropFrame.offsetWidth,
      height: targetCropFrame.offsetHeight
    };
    e.preventDefault();
    e.stopPropagation();
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDraggingFrame && !isResizingFrame) return;

    const imgW = previewImg.clientWidth || previewViewport.offsetWidth || 380;
    const imgH = previewImg.clientHeight || previewViewport.offsetHeight || 380;
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;

    if (isDraggingFrame) {
      let newLeft = Math.max(0, Math.min(imgW - initialFrameRect.width, initialFrameRect.left + dx));
      let newTop = Math.max(0, Math.min(imgH - initialFrameRect.height, initialFrameRect.top + dy));

      targetCropFrame.style.left = `${newLeft}px`;
      targetCropFrame.style.top = `${newTop}px`;
    } else if (isResizingFrame) {
      let l = initialFrameRect.left;
      let t = initialFrameRect.top;
      let w = initialFrameRect.width;
      let h = initialFrameRect.height;

      if (activeHandle.includes('r')) w = Math.max(30, Math.min(imgW - l, initialFrameRect.width + dx));
      if (activeHandle.includes('b')) h = Math.max(30, Math.min(imgH - t, initialFrameRect.height + dy));
      if (activeHandle.includes('l')) {
        const potentialL = Math.max(0, Math.min(initialFrameRect.left + initialFrameRect.width - 30, initialFrameRect.left + dx));
        w = initialFrameRect.width + (initialFrameRect.left - potentialL);
        l = potentialL;
      }
      if (activeHandle.includes('t')) {
        const potentialT = Math.max(0, Math.min(initialFrameRect.top + initialFrameRect.height - 30, initialFrameRect.top + dy));
        h = initialFrameRect.height + (initialFrameRect.top - potentialT);
        t = potentialT;
      }

      targetCropFrame.style.left = `${l}px`;
      targetCropFrame.style.top = `${t}px`;
      targetCropFrame.style.width = `${w}px`;
      targetCropFrame.style.height = `${h}px`;
    }
  });

  window.addEventListener('mouseup', () => {
    if (isDraggingFrame || isResizingFrame) {
      isDraggingFrame = false;
      isResizingFrame = false;
      activeHandle = null;

      const imgW = previewImg.clientWidth || previewViewport.offsetWidth || 380;
      const imgH = previewImg.clientHeight || previewViewport.offsetHeight || 380;

      const l = targetCropFrame.offsetLeft / imgW;
      const t = targetCropFrame.offsetTop / imgH;
      const r = (targetCropFrame.offsetLeft + targetCropFrame.offsetWidth) / imgW;
      const b = (targetCropFrame.offsetTop + targetCropFrame.offsetHeight) / imgH;

      activeCropBox = {
        x1: Math.max(0, Math.min(1, l)),
        y1: Math.max(0, Math.min(1, t)),
        x2: Math.max(0, Math.min(1, r)),
        y2: Math.max(0, Math.min(1, b))
      };

      cropBadge.textContent = '🎯 Custom Target Frame';
      document.querySelectorAll('.hotspot-pin, .hotspot-chip-btn').forEach(el => el.classList.remove('active'));

      // Automatically re-run search on adjusted frame with AI driving new results
      executeSearch(true);
    }
  });

  // ==========================================================================
  // HUD FILTER UPDATES
  // ==========================================================================
  function updateHud(analysis, uploadedMeta) {
    isUpdatingHud = true;
    recognitionHud.style.display = 'block';

    const meta = uploadedMeta || {};
    const shape = meta.diamond_shape || analysis.diamond_shape || analysis.detected_shape || 'Emerald';
    const metal = meta.band_metal_tone || analysis.band_metal_tone || analysis.detected_band_color || 'Yellow Gold';
    const arch = meta.band_architecture || analysis.band_architecture || analysis.detected_band_architecture || 'Tapered Shank';
    const prong = meta.prong_setting || analysis.prong_setting || analysis.detected_prong_style || 'Hidden Halo';
    const style = meta.ring_style || analysis.ring_style || analysis.detected_style || 'Solitaire';

    setSelectValue(hudShapeSelect, shape);
    setSelectValue(hudMetalSelect, metal);
    setSelectValue(hudArchSelect, arch);
    setSelectValue(hudProngSelect, prong);
    setSelectValue(hudStyleSelect, style);

    isUpdatingHud = false;
  }

  function setSelectValue(selectEl, value) {
    if (!value || !selectEl) return;
    const valClean = value.trim().toLowerCase();

    // 1. Exact match
    for (let opt of selectEl.options) {
      if (opt.value && opt.value.toLowerCase() === valClean) {
        selectEl.value = opt.value;
        return;
      }
    }

    // 2. Meaningful keyword substring match
    for (let opt of selectEl.options) {
      if (opt.value) {
        const optClean = opt.value.toLowerCase();
        if (valClean.includes(optClean) || optClean.includes(valClean)) {
          selectEl.value = opt.value;
          return;
        }
      }
    }

    // 3. Fallback: Add option dynamically and select it
    const newOpt = new Option(value, value, true, true);
    selectEl.add(newOpt);
    selectEl.value = value;
  }

  // ==========================================================================
  // PRODUCT RESULTS RENDERING
  // ==========================================================================
  function renderResults(products, analysis) {
    searchLoading.style.display = 'none';
    searchEmpty.style.display = 'none';
    resultsGrid.style.display = 'grid';
    const resultsCountText = document.getElementById('results-count-text');
    if (resultsCountText && analysis) {
      resultsCountText.textContent = `Found ${products.length} matching rings (${analysis.detected_shape || 'Diamond'} Cut in ${analysis.detected_band_color || 'Gold'})`;
    } else if (resultsCountText) {
      resultsCountText.textContent = `Found ${products.length} matching rings`;
    }

    if (products.length === 0) {
      resultsGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1; padding: 3rem; text-align: center;">
          <p>No rings found matching the exact filters. Try adjusting target crop frame or clear some attribute filters.</p>
        </div>
      `;
      return;
    }

    resultsGrid.innerHTML = products.map((item, idx) => {
      const metalImgs = item.metalImages || {};
      const whiteImg = metalImgs['14K White Gold'] || item.image;
      const yellowImg = metalImgs['14K Yellow Gold'] || item.image;
      const roseImg = metalImgs['14K Rose Gold'] || item.image;

      const videoUrl = (item.videos && item.videos.length > 0) ? item.videos[0] : (item.default_video || '');
      const hasVideo = !!videoUrl;
      const meta = item.metadata || {};

      return `
        <div class="product-card" data-index="${idx}">
          <div class="card-img-container">
            <img class="card-primary-img" src="${item.image}" alt="${item.productFullName}" data-default="${item.image}" loading="lazy" decoding="async">
            <div class="similarity-pill">${item.similarity_score}% Match</div>
            
            ${hasVideo ? `
              <button class="view-360-btn" data-video="${videoUrl}" data-title="${item.productFullName}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                360° View
              </button>
            ` : ''}
          </div>

          <div class="card-body">
            <div class="metal-swatches">
              <span class="swatch white active" title="14K White Gold" data-img="${whiteImg}"></span>
              <span class="swatch yellow" title="14K Yellow Gold" data-img="${yellowImg}"></span>
              <span class="swatch rose" title="14K Rose Gold" data-img="${roseImg}"></span>
            </div>

            <h3 class="card-title" title="${item.productFullName}">${item.productFullName}</h3>

            <!-- Metadata Quick Badges -->
            <div class="card-meta-chips">
              <span class="meta-chip-pill gold">💎 ${item.shape}</span>
              <span class="meta-chip-pill">✨ ${item.total_carat_weight || '1.5'} CTW</span>
              <span class="meta-chip-pill">🏷️ SKU: ${item.sku || item.id}</span>
            </div>

            <!-- Metadata Collapsible Specs Toggle -->
            <button class="card-metadata-toggle-btn" data-target="drawer-${idx}">
              <span>📋 Product Metadata & Specs</span>
              <span>▾</span>
            </button>

            <!-- Collapsible Metadata Drawer -->
            <div class="card-metadata-drawer" id="drawer-${idx}">
              <div class="meta-spec-grid">
                <div class="meta-spec-item">
                  <span class="meta-k">Shape</span>
                  <span class="meta-v">${item.shape}</span>
                </div>
                <div class="meta-spec-item">
                  <span class="meta-k">Color</span>
                  <span class="meta-v">${item.color || item.metalType}</span>
                </div>
                <div class="meta-spec-item">
                  <span class="meta-k">Band Type</span>
                  <span class="meta-v">${item.band_type || item.bandType || 'Plain Solitaire Band'}</span>
                </div>
                <div class="meta-spec-item">
                  <span class="meta-k">Band Color</span>
                  <span class="meta-v">${item.band_color || item.metalType}</span>
                </div>
                <div class="meta-spec-item">
                  <span class="meta-k">Band Configuration</span>
                  <span class="meta-v">${item.band_configuration || item.bandArchitecture || 'Classic Straight Shank'}</span>
                </div>
                <div class="meta-spec-item">
                  <span class="meta-k">Prong Color</span>
                  <span class="meta-v">${item.prong_color || item.color || item.metalType}</span>
                </div>
                <div class="meta-spec-item">
                  <span class="meta-k">Prong Style</span>
                  <span class="meta-v">${item.prong_style || item.settingType || 'Classic 4-Prong Setting'}</span>
                </div>
                <div class="meta-spec-item">
                  <span class="meta-k">Total Carat Weight</span>
                  <span class="meta-v">${item.total_carat_weight || '1.5'} CTW</span>
                </div>
              </div>
            </div>

            <div class="card-footer">
              <div class="card-price">$${(item.price || 1450).toLocaleString()}</div>
              <a href="${item.pdpUrl || '#'}" target="_blank" class="view-pdp-link">
                View Details →
              </a>
            </div>
          </div>
        </div>
      `;
    }).join('');

    // Attach metadata drawer toggles
    document.querySelectorAll('.card-metadata-toggle-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const targetId = btn.getAttribute('data-target');
        const drawer = document.getElementById(targetId);
        if (drawer) {
          drawer.classList.toggle('open');
          const isOpened = drawer.classList.contains('open');
          btn.querySelector('span:last-child').textContent = isOpened ? '▴' : '▾';
        }
      });
    });

    // Attach swatch switchers
    document.querySelectorAll('.product-card').forEach(card => {
      const primaryImg = card.querySelector('.card-primary-img');
      const swatches = card.querySelectorAll('.swatch');

      swatches.forEach(swatch => {
        swatch.addEventListener('click', (e) => {
          e.stopPropagation();
          swatches.forEach(s => s.classList.remove('active'));
          swatch.classList.add('active');

          const newImg = swatch.getAttribute('data-img');
          if (newImg && newImg !== primaryImg.src) {
            primaryImg.style.opacity = '0.4';
            primaryImg.src = newImg;
            primaryImg.onload = () => {
              primaryImg.style.opacity = '1';
            };
          }
        });
      });
    });

    // Attach 360 Video Triggers
    document.querySelectorAll('.view-360-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const vUrl = btn.getAttribute('data-video');
        const vTitle = btn.getAttribute('data-title');
        openVideoModal(vUrl, vTitle);
      });
    });
  }

  function openVideoModal(url, title) {
    if (!url) return;
    modalTitle.textContent = title || '360° Diamond Ring View';
    modalVideo.src = url;
    videoModal.classList.add('open');
    modalVideo.play();
  }

  modalClose.addEventListener('click', closeVideoModal);
  videoModal.addEventListener('click', (e) => {
    if (e.target === videoModal) closeVideoModal();
  });

  function closeVideoModal() {
    videoModal.classList.remove('open');
    modalVideo.pause();
    modalVideo.src = '';
  }

  function showLoading() {
    searchEmpty.style.display = 'none';
    resultsGrid.style.display = 'none';
    searchLoading.style.display = 'flex';
  }

  function showEmpty() {
    searchLoading.style.display = 'none';
    resultsGrid.style.display = 'none';
    searchEmpty.style.display = 'flex';
  }

  // Tabs
  tabCardsBtn.addEventListener('click', () => {
    tabCardsBtn.classList.add('active');
    tabRawBtn.classList.remove('active');
    resultsGrid.style.display = lastResultData ? 'grid' : 'none';
    jsonViewer.style.display = 'none';
  });

  tabRawBtn.addEventListener('click', () => {
    tabRawBtn.classList.add('active');
    tabCardsBtn.classList.remove('active');
    resultsGrid.style.display = 'none';
    jsonViewer.style.display = 'block';
  });

  copyJsonAction.addEventListener('click', () => {
    navigator.clipboard.writeText(jsonOutputText.textContent);
    copyJsonAction.textContent = 'Copied!';
    setTimeout(() => {
      copyJsonAction.textContent = 'Copy JSON';
    }, 2000);
  });
});
