/**
 * SpiderNet Results Viewer - Main JavaScript
 */

$(document).ready(function() {
    // Initialize image modal functionality
    initImageModal();

    // Initialize lazy loading
    initLazyLoading();

    // Initialize keyboard navigation
    initKeyboardNav();
});


/**
 * Initialize image modal for plot enlargement
 */
function initImageModal() {
    $('.plot-image, .plot-overlay').on('click', function(e) {
        e.preventDefault();
        openImageModal($(this));
    });

    // Add keyboard accessibility
    $('.plot-image').on('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openImageModal($(this));
        }
    });

    // Make images keyboard focusable
    $('.plot-image').attr('tabindex', '0').attr('role', 'button');
}

/**
 * Open image modal with given image
 */
function openImageModal($element) {
    // Get the image element
    const $img = $element.hasClass('plot-image') ? $element : $element.closest('.plot-image-container').find('.plot-image');

    const imageSrc = $img.attr('src');
    const imageTitle = $img.data('title');
    const pdfAvailable = $img.data('pdf-available');
    const pdfPath = $img.data('pdf-path');

    // Set modal content
    $('#modalImage').attr('src', imageSrc).attr('alt', imageTitle);
    $('#imageModalLabel').text(imageTitle);

    // Handle PDF download button
    if (pdfAvailable === true || pdfAvailable === 'True') {
        const pdfUrl = imageSrc.replace('.png', '.pdf').replace(/\.png$/, '.pdf');
        $('#downloadPdf').attr('href', pdfUrl).show();
    } else {
        $('#downloadPdf').hide();
    }

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();
}


/**
 * Initialize lazy loading for images
 */
function initLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src || img.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        document.querySelectorAll('img.lazy').forEach(img => {
            imageObserver.observe(img);
        });
    }
}


/**
 * Initialize keyboard navigation for modal
 */
function initKeyboardNav() {
    $(document).on('keydown', function(e) {
        // Check if modal is open
        const modal = document.getElementById('imageModal');
        if (!modal || !modal.classList.contains('show')) {
            return;
        }

        // ESC key - close modal
        if (e.key === 'Escape') {
            bootstrap.Modal.getInstance(modal).hide();
        }

        // Arrow keys - navigate between images (future enhancement)
        // if (e.key === 'ArrowLeft') {
        //     // Previous image
        // }
        // if (e.key === 'ArrowRight') {
        //     // Next image
        // }
    });
}


/**
 * Show loading indicator
 */
function showLoading(container) {
    const $container = $(container);
    $container.html('<div class="loading"></div>');
}


/**
 * Hide loading indicator
 */
function hideLoading(container) {
    const $container = $(container);
    $container.find('.loading').remove();
}


/**
 * Handle image load errors
 */
$(document).on('error', '.plot-image', function() {
    $(this).attr('src', 'data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22400%22%20height%3D%22300%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Crect%20fill%3D%22%23f8f9fa%22%20width%3D%22400%22%20height%3D%22300%22%2F%3E%3Ctext%20fill%3D%22%236c757d%22%20font-family%3D%22sans-serif%22%20font-size%3D%2214%22%20x%3D%22150%22%20y%3D%22150%22%3EImage%20not%20found%3C%2Ftext%3E%3C%2Fsvg%3E');
});


/**
 * Smooth scroll to section
 */
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}


/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Success - could add toast notification here
    }).catch(err => {
        // Handle error silently in production
    });
}


/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}


/**
 * Debounce function for search/filter
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}


/**
 * Filter plots by keyword (for future search feature)
 */
function filterPlots(keyword) {
    const $plots = $('.plot-card');
    const searchTerm = keyword.toLowerCase();

    $plots.each(function() {
        const $plot = $(this);
        const title = $plot.find('.plot-title').text().toLowerCase();

        if (title.includes(searchTerm)) {
            $plot.show();
        } else {
            $plot.hide();
        }
    });
}


// Export functions for global use
window.SpiderNetViewer = {
    showLoading,
    hideLoading,
    scrollToSection,
    copyToClipboard,
    formatNumber,
    filterPlots
};
