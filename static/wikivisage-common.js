var VIDEO_EXTENSIONS = ['.webm', '.ogv', '.ogg'];
var TIF_EXTENSIONS = ['.tif', '.tiff'];
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
    // Spark MD5 not available — use the commons-thumb endpoint for non-standard files
    var ext = '.' + clean.split('.').pop().toLowerCase();
    if (VIDEO_EXTENSIONS.indexOf(ext) !== -1 || TIF_EXTENSIONS.indexOf(ext) !== -1 || ext === '.svg') {
        return '/commons-thumb/' + encodeURIComponent(clean) + '?width=' + width;
    }
    return 'https://commons.wikimedia.org/wiki/Special:FilePath/' + encodeURIComponent(clean) + '?width=' + width;
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
        if (!r.ok) throw new Error('Network response was not ok');
        return r.json();
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