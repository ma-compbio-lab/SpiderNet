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

    // Initialize search and filter functionality
    initSearchAndFilter();
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
 * Initialize keyboard navigation for modal and global shortcuts
 */
function initKeyboardNav() {
    $(document).on('keydown', function(e) {
        // Check if modal is open
        const modal = document.getElementById('imageModal');
        const isModalOpen = modal && modal.classList.contains('show');

        // Ignore if user is typing in an input
        if ($(e.target).is('input, textarea')) {
            if (e.key === 'Escape') {
                e.target.blur();
            }
            return;
        }

        // Global shortcuts (work everywhere)
        if (e.key === '?' && !isModalOpen) {
            e.preventDefault();
            showKeyboardHelp();
            return;
        }

        if ((e.key === '/' || e.key.toLowerCase() === 's') && !isModalOpen) {
            e.preventDefault();
            const searchBox = document.getElementById('plotSearch');
            if (searchBox) {
                searchBox.focus();
            }
            return;
        }

        // Modal-specific shortcuts
        if (isModalOpen) {
            // ESC key - close modal
            if (e.key === 'Escape') {
                bootstrap.Modal.getInstance(modal).hide();
                return;
            }

            // F key - toggle fullscreen
            if (e.key.toLowerCase() === 'f') {
                e.preventDefault();
                const img = modal.querySelector('#modalImage');
                if (img && img.requestFullscreen) {
                    if (document.fullscreenElement) {
                        document.exitFullscreen();
                    } else {
                        img.requestFullscreen();
                    }
                }
                return;
            }

            // Arrow keys - navigate between images (future enhancement)
            // if (e.key === 'ArrowLeft') {
            //     // Previous image
            // }
            // if (e.key === 'ArrowRight') {
            //     // Next image
            // }
        }
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
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Close dropdown after clicking
        const dropdowns = document.querySelectorAll('.dropdown-menu.show');
        dropdowns.forEach(dropdown => {
            const bsDropdown = bootstrap.Dropdown.getInstance(dropdown.previousElementSibling);
            if (bsDropdown) bsDropdown.hide();
        });
    }
}

// Make scrollToSection globally accessible
window.scrollToSection = scrollToSection;


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
 * Initialize search and filter functionality
 */
function initSearchAndFilter() {
    const $plotSearch = $('#plotSearch');
    const $filterSection = $('#filterSection');
    const $sortPlots = $('#sortPlots');
    const $clearSearch = $('#clearSearch');
    const $searchResultsText = $('#searchResultsText');

    if (!$plotSearch.length) return; // Exit if search panel not present

    // Collect all plots data for searching
    const plots = [];
    $('.plot-card').each(function(idx) {
        const $card = $(this);
        const $section = $card.closest('.section-container');
        const sectionId = $section.attr('id') || '';

        plots.push({
            element: this,
            columnElement: $card.parent()[0], // Store the parent column div
            title: $card.find('.plot-title').text().toLowerCase(),
            description: $card.find('.plot-description').text().toLowerCase(),
            section: $section.find('.section-title').text().trim(),
            sectionId: sectionId
        });
    });

    // Search functionality
    $plotSearch.on('input', debounce(function() {
        performSearch();
    }, 300));

    // Section filter
    $filterSection.on('change', function() {
        performSearch();
    });

    // Sort functionality
    $sortPlots.on('change', function() {
        performSort($(this).val());
    });

    // Clear button
    $clearSearch.on('click', function() {
        $plotSearch.val('');
        $filterSection.val('');
        $sortPlots.val('default');
        performSearch();
    });

    // Perform search
    function performSearch() {
        const query = $plotSearch.val().toLowerCase();
        const sectionFilter = $filterSection.val();
        let visibleCount = 0;
        let totalCount = plots.length;

        // First, reset visibility: show all sections and columns
        $('.section-container').show();
        $('.col-md-6, .col-lg-4, .col-xl-3').show();

        plots.forEach(plot => {
            const matchesQuery = query === '' ||
                plot.title.includes(query) ||
                plot.description.includes(query);

            const matchesSection = sectionFilter === '' ||
                plot.sectionId === sectionFilter;

            if (matchesQuery && matchesSection) {
                $(plot.columnElement).show(); // Show the column div
                visibleCount++;
            } else {
                $(plot.columnElement).hide(); // Hide the column div
            }
        });

        // Update results text
        if (query || sectionFilter) {
            $searchResultsText.html(`Found <strong>${visibleCount}</strong> of ${totalCount} plots`);
        } else {
            $searchResultsText.text('All plots shown');
        }

        // Hide empty sections - check column visibility
        $('.section-container').each(function() {
            const $section = $(this);
            const visibleColumns = $section.find('.col-md-6:visible, .col-lg-4:visible, .col-xl-3:visible').length;
            if (visibleColumns > 0) {
                $section.show();
            } else {
                $section.hide();
            }
        });
    }

    // Perform sort
    function performSort(sortType) {
        $('.section-container').each(function() {
            const $section = $(this);
            const $plotsContainer = $section.find('.row.g-3');
            const $plots = $plotsContainer.find('.col-md-6, .col-lg-4, .col-xl-3');

            const sortedPlots = $plots.sort(function(a, b) {
                const titleA = $(a).find('.plot-title').text();
                const titleB = $(b).find('.plot-title').text();

                switch(sortType) {
                    case 'name-asc':
                        return titleA.localeCompare(titleB);
                    case 'name-desc':
                        return titleB.localeCompare(titleA);
                    default:
                        return 0; // Keep original order
                }
            });

            $plotsContainer.append(sortedPlots);
        });
    }
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


/**
 * Copy plot title to clipboard
 */
function copyPlotTitle(button, title) {
    navigator.clipboard.writeText(title).then(() => {
        // Visual feedback
        const $btn = $(button);
        const originalHTML = $btn.html();
        $btn.html('<i class="fas fa-check"></i>');
        $btn.addClass('text-success');

        setTimeout(() => {
            $btn.html(originalHTML);
            $btn.removeClass('text-success');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}


/**
 * Download plot image
 */
function downloadPlot(url, title) {
    const link = document.createElement('a');
    link.href = url;
    link.download = title.replace(/[^a-z0-9]/gi, '_').toLowerCase() + '.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}


/**
 * Copy plot link to clipboard
 */
function copyPlotLink(button, plotPath) {
    const url = window.location.origin + window.location.pathname + '#' + encodeURIComponent(plotPath);

    navigator.clipboard.writeText(url).then(() => {
        // Visual feedback
        const $btn = $(button);
        $btn.find('i').removeClass('fa-link').addClass('fa-check');
        $btn.addClass('text-success');

        setTimeout(() => {
            $btn.find('i').removeClass('fa-check').addClass('fa-link');
            $btn.removeClass('text-success');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy link:', err);
    });
}


/**
 * Share plot using Web Share API or fallback
 */
function sharePlot(title, path) {
    const url = window.location.origin + window.location.pathname + '#' + encodeURIComponent(path);

    if (navigator.share) {
        navigator.share({
            title: `SpiderNet Plot: ${title}`,
            text: `Check out this SpiderNet analysis plot: ${title}`,
            url: url
        }).catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Share failed:', err);
            }
        });
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(url).then(() => {
            alert('Link copied to clipboard!');
        });
    }
}


/**
 * Show keyboard shortcuts help
 */
function showKeyboardHelp() {
    const helpHTML = `
        <div class="keyboard-shortcuts-help">
            <h5 class="mb-3"><i class="fas fa-keyboard me-2"></i>Keyboard Shortcuts</h5>
            <table class="table table-sm">
                <tbody>
                    <tr>
                        <td><kbd>?</kbd></td>
                        <td>Show this help</td>
                    </tr>
                    <tr>
                        <td><kbd>Esc</kbd></td>
                        <td>Close modal or dialog</td>
                    </tr>
                    <tr>
                        <td><kbd>← →</kbd></td>
                        <td>Navigate between plots (in modal)</td>
                    </tr>
                    <tr>
                        <td><kbd>/</kbd> or <kbd>S</kbd></td>
                        <td>Focus search box</td>
                    </tr>
                    <tr>
                        <td><kbd>F</kbd></td>
                        <td>Toggle fullscreen (when image open)</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    // Show in modal (reusing image modal)
    $('#imageModalLabel').html('<i class="fas fa-keyboard me-2"></i>Keyboard Shortcuts');
    $('#modalImage').replaceWith(helpHTML);
    $('#downloadPdf').hide();

    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();

    // Restore modal on close
    $('#imageModal').one('hidden.bs.modal', function() {
        $('.keyboard-shortcuts-help').replaceWith('<img id="modalImage" src="" class="img-fluid" alt="Plot">');
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

// Make action functions globally available
window.copyPlotTitle = copyPlotTitle;
window.downloadPlot = downloadPlot;
window.copyPlotLink = copyPlotLink;
window.sharePlot = sharePlot;
window.showKeyboardHelp = showKeyboardHelp;
