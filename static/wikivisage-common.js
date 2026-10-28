var THUMB_STEPS = [20, 40, 60, 120, 250, 330, 500, 960, 1280, 1920, 3840];
function snapThumbWidth(w) {
    for (var i = 0; i < THUMB_STEPS.length; i++) {
        if (THUMB_STEPS[i] >= w) return THUMB_STEPS[i];
    }
    return THUMB_STEPS[THUMB_STEPS.length - 1];
}

function commonsThumbUrl(fileTitle, width) {
    width = snapThumbWidth(width);
    var clean = fileTitle.replace('File:', '').replace(/ /g, '_');
    return '/commons-thumb/' + encodeURIComponent(clean) + '?width=' + width;
}

/**
 * POST a FormData request with CSRF token. Returns a Promise that resolves
 * to the parsed JSON response body. Rejects on network error or non-2xx status.
 *
 * @param {string} url - The endpoint URL.
 * @param {string} csrfToken - CSRF token value.
 * @param {Object} fields - Key/value pairs to append to the FormData.
 * @returns {Promise<Object>} Parsed JSON response.
 */
function postForm(url, csrfToken, fields) {
    var formData = new FormData();
    formData.append('csrf_token', csrfToken);
    for (var key in fields) {
        if (fields.hasOwnProperty(key)) formData.append(key, fields[key]);
    }
    return fetch(url, { method: 'POST', body: formData }).then(function(r) {
        return r.json().catch(function() {
            return { error: 'Server error (HTTP ' + r.status + ')' };
        }).then(function(data) {
            if (!r.ok) {
                var err = new Error(data.error || 'Network response was not ok');
                err.data = data;
                err.status = r.status;
                throw err;
            }
            return data;
        });
    });
}

/**
 * Convert a detection-space bounding box {top, right, bottom, left} to
 * display-space pixel rectangle {left, top, width, height}.
 *
 * @param {{top:number, right:number, bottom:number, left:number}} bbox
 * @param {number} scaleX - Display pixels per detection pixel (horizontal).
 * @param {number} scaleY - Display pixels per detection pixel (vertical).
 * @param {number} offsetX - Horizontal offset of the image within its container.
 * @param {number} offsetY - Vertical offset of the image within its container.
 * @returns {{left:number, top:number, width:number, height:number}}
 */
function bboxToDisplayRect(bbox, scaleX, scaleY, offsetX, offsetY) {
    return {
        left: bbox.left * scaleX + offsetX,
        top: bbox.top * scaleY + offsetY,
        width: (bbox.right - bbox.left) * scaleX,
        height: (bbox.bottom - bbox.top) * scaleY
    };
}

/**
 * Convert a display-space pixel rectangle back to a detection-space bounding
 * box with rounded integer coordinates.
 *
 * @param {{left:number, top:number, width:number, height:number}} rect
 * @param {number} scaleX - Display pixels per detection pixel (horizontal).
 * @param {number} scaleY - Display pixels per detection pixel (vertical).
 * @param {number} offsetX - Horizontal offset of the image within its container.
 * @param {number} offsetY - Vertical offset of the image within its container.
 * @returns {{top:number, right:number, bottom:number, left:number}}
 */
function displayRectToDetectionBbox(rect, scaleX, scaleY, offsetX, offsetY) {
    return {
        top: Math.round((rect.top - offsetY) / scaleY),
        right: Math.round((rect.left + rect.width - offsetX) / scaleX),
        bottom: Math.round((rect.top + rect.height - offsetY) / scaleY),
        left: Math.round((rect.left - offsetX) / scaleX)
    };
}

/**
 * Highest device-pixels-per-source-pixel the zoom into the face may magnify a
 * gallery thumbnail to. Above this the face looks pixelated, so computeFaceView()
 * shows more surroundings (or asks for a larger source) instead. It limits the
 * zoom only: an original so small that it is shown whole (contain mode) can still
 * be magnified past it to fill the card. The one place to tune it.
 */
var MAX_DEVICE_UPSCALE = 2.0;
var FACE_VIEW_PAD = 0.8;               // crop = box + this x box size on each side
var FACE_VIEW_MIN_ZOOM = 1.1;          // below this zoom over the whole image, show it uncropped
var FACE_VIEW_MAX_ASPECT_DRIFT = 0.03; // natural vs detection aspect tolerance
var FACE_VIEW_MARKER_MAX_COVER = 0.7;  // hide the marker when the box fills this much of the card
// Widest source the progressive upgrade asks for. Keep equal to _GALLERY_HIRES_MAX in app.py,
// which decides which direct thumbnail URLs api_gallery sends along with each row.
var FACE_VIEW_MAX_SOURCE_WIDTH = 960;

/**
 * Largest standard Commons thumbnail step that is <= w (0 when none).
 */
function largestThumbStepAtMost(w) {
    var best = 0;
    for (var i = 0; i < THUMB_STEPS.length; i++) {
        if (THUMB_STEPS[i] <= w) best = THUMB_STEPS[i];
    }
    return best;
}

/**
 * Width of a larger Commons thumbnail worth requesting, or 0 for "keep the
 * current one". Commons refuses thumbnails wider than the original, and a
 * detection width below 960 means the original is exactly that wide, so the
 * result never exceeds min(960, largest step <= detW).
 *
 * @param {number} needNatW - Natural source width the desired view needs.
 * @param {number} natW - Natural width of the image currently shown.
 * @param {number} detW - Detection-space width (the original's width when < 960).
 * @returns {number}
 */
function faceViewSourceWidth(needNatW, natW, detW) {
    if (!(needNatW > natW) || !(detW > 0)) return 0;
    var limit = Math.min(FACE_VIEW_MAX_SOURCE_WIDTH, largestThumbStepAtMost(detW));
    var w = Math.min(snapThumbWidth(needNatW), limit);
    return w > natW ? w : 0;
}

/**
 * Geometry of a gallery thumbnail: where to place the image inside its card
 * and where the classified face sits. Pure (no DOM), all lengths in CSS px of
 * the container.
 *
 * Returns null when the geometry cannot be trusted (missing detection size,
 * degenerate box, natural/detection aspect mismatch such as an EXIF-rotated
 * original): show the plain contained image and no marker.
 *
 * @param {{top:number, right:number, bottom:number, left:number}} bbox - detection space
 * @param {number} detW - Detection image width.
 * @param {number} detH - Detection image height.
 * @param {number} natW - Natural width of the loaded image.
 * @param {number} natH - Natural height of the loaded image.
 * @param {number} containerW
 * @param {number} containerH
 * @param {number} dpr - Device pixel ratio.
 * @param {{zoom?:boolean, maxDeviceUpscale?:number}} [opts] - zoom:false forces contain mode.
 * @returns {null | {mode:'crop'|'contain', imgLeft:number, imgTop:number, imgWidth:number,
 *                   imgHeight:number, box:{left:number, top:number, width:number, height:number},
 *                   markerVisible:boolean, needNatW:number}}
 *   needNatW is the natural source width at which the desired crop needs at
 *   most the capped upscale. It is 0 when zoom is off and when the crop would be
 *   below FACE_VIEW_MIN_ZOOM (a larger source would not change that).
 */
function computeFaceView(bbox, detW, detH, natW, natH, containerW, containerH, dpr, opts) {
    opts = opts || {};
    var maxUp = opts.maxDeviceUpscale || MAX_DEVICE_UPSCALE;
    dpr = dpr > 0 ? dpr : 1;
    if (!(detW > 0 && detH > 0 && natW > 0 && natH > 0 && containerW > 0 && containerH > 0)) return null;
    if (!bbox) return null;

    var faceW = bbox.right - bbox.left;
    var faceH = bbox.bottom - bbox.top;
    if (!(faceW > 0 && faceH > 0)) return null;
    if (Math.abs((natW / natH) / (detW / detH) - 1) > FACE_VIEW_MAX_ASPECT_DRIFT) return null;

    var result = { needNatW: 0 };

    function finish(mode, imgLeft, imgTop, imgWidth, imgHeight) {
        var box = bboxToDisplayRect(bbox, imgWidth / detW, imgHeight / detH, imgLeft, imgTop);
        result.mode = mode;
        result.imgLeft = imgLeft;
        result.imgTop = imgTop;
        result.imgWidth = imgWidth;
        result.imgHeight = imgHeight;
        result.box = box;
        result.markerVisible = !(box.width > FACE_VIEW_MARKER_MAX_COVER * containerW &&
                                 box.height > FACE_VIEW_MARKER_MAX_COVER * containerH);
        return result;
    }

    function contain() {
        var s = Math.min(containerW / natW, containerH / natH);
        var w = natW * s;
        var h = natH * s;
        return finish('contain', (containerW - w) / 2, (containerH - h) / 2, w, h);
    }

    if (opts.zoom === false) return contain();

    // Desired crop (detection space): the box plus padding, clamped to the image.
    var cropLeft = Math.max(0, bbox.left - faceW * FACE_VIEW_PAD);
    var cropTop = Math.max(0, bbox.top - faceH * FACE_VIEW_PAD);
    var cropRight = Math.min(detW, bbox.right + faceW * FACE_VIEW_PAD);
    var cropBottom = Math.min(detH, bbox.bottom + faceH * FACE_VIEW_PAD);
    var cropW = cropRight - cropLeft;
    var cropH = cropBottom - cropTop;
    if (!(cropW > 0 && cropH > 0)) return null;

    // CSS px per detection px needed for the crop to cover the card, and the
    // natural source width at which that stays within the upscale cap.
    var wantedPerDet = Math.max(containerW / cropW, containerH / cropH);
    result.needNatW = Math.ceil(detW * wantedPerDet * dpr / maxUp);

    var wantedScale = wantedPerDet * detW / natW;     // CSS px per natural px
    var capScale = maxUp / dpr;
    var fullCover = Math.max(containerW / natW, containerH / natH);
    // wantedScale / fullCover does not depend on the source width (the aspects match within the
    // drift tolerance), so a larger source cannot turn this card into a crop: ask for none.
    if (wantedScale / fullCover < FACE_VIEW_MIN_ZOOM) {
        result.needNatW = 0;
        return contain();
    }
    // Here a larger source lowers fullCover and can enable the crop, so needNatW stays set.
    if (capScale < fullCover) return contain();

    var scale = Math.min(wantedScale, capScale);
    if (scale / fullCover < FACE_VIEW_MIN_ZOOM) return contain();

    var imgWidth = natW * scale;
    var imgHeight = natH * scale;
    var cx = (cropLeft + cropW / 2) * imgWidth / detW;
    var cy = (cropTop + cropH / 2) * imgHeight / detH;
    var imgLeft = Math.min(0, Math.max(containerW - imgWidth, containerW / 2 - cx));
    var imgTop = Math.min(0, Math.max(containerH - imgHeight, containerH / 2 - cy));
    return finish('crop', imgLeft, imgTop, imgWidth, imgHeight);
}
