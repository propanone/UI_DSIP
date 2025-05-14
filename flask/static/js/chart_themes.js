// static/js/chart_themes.js
// Content from your provided chart_themes.js
/**
 * chart_themes.js
 * Provides Chart.js configuration options styled to match the Shadcn UI aesthetic,
 * dynamically reading CSS variables for theme colors (best effort).
 */

/**
 * Safely gets a computed CSS variable value as an HSL string.
 * @param {string} varName -- The CSS variable name (e.g., '--foreground')
 * @param {string} fallbackColor - A fallback HSL color string if the variable is not found.
 * @returns {string} The HSL color string (e.g., 'hsl(240 10% 3.9%)') or fallback.
 */
function getCssVariableValue(varName, fallbackColor) {
    try {
        const value = getComputedStyle(document.documentElement).getPropertyValue(varName)?.trim();
        if (value) {
            // Assuming the value is in the format 'H S% L%' or 'H S L'
            // Convert space-separated HSL to hsl(H, S%, L%) format if needed,
            // although Chart.js v3+ often handles space-separated HSL.
            // For maximum compatibility, let's format it explicitly.
            const parts = value.split(' ');
            if (parts.length === 3) {
                return `hsl(${parts[0]}, ${parts[1].endsWith('%') ? parts[1] : parts[1] + '%'}, ${parts[2].endsWith('%') ? parts[2] : parts[2] + '%'})`;
            }
            return value; // Return as is if not HSL parts
        }
    } catch (e) {
        console.warn(`[ChartTheme] Error reading CSS variable ${varName}:`, e);
    }
    console.warn(`[ChartTheme] CSS variable ${varName} not found, using fallback ${fallbackColor}.`);
    return fallbackColor;
}

/**
 * Generates a Chart.js options object themed like Shadcn UI.
 * Dynamically reads CSS variables for colors.
 * @param {boolean} [isDarkModeOverride] - Optional: Force dark/light mode.
 * @returns {object} A Chart.js options object.
 */
function getChartJsShadcnOptions(isDarkModeOverride = null) {
    const isDarkMode = isDarkModeOverride ?? document.documentElement.classList.contains('dark');
    console.log(`[ChartTheme] Generating options. isDarkMode detected: ${isDarkMode}`);

    // Define fallbacks based on standard Shadcn HSL values
    const fallbackGrid = isDarkMode ? 'hsl(240, 3.7%, 15.9%)' : 'hsl(240, 5.9%, 90%)';      // --border
    const fallbackTick = isDarkMode ? 'hsl(240, 5%, 64.9%)' : 'hsl(240, 3.8%, 46.1%)';      // --muted-foreground
    const fallbackTitle = isDarkMode ? 'hsl(0, 0%, 98%)' : 'hsl(240, 10%, 3.9%)';           // --foreground
    const fallbackTooltipBg = isDarkMode ? 'hsl(240, 10%, 3.9%)' : 'hsl(0, 0%, 100%)';       // --popover / --card
    const fallbackTooltipFore = isDarkMode ? 'hsl(0, 0%, 98%)' : 'hsl(240, 10%, 3.9%)';    // --popover-foreground / --card-foreground

    const gridColor = getCssVariableValue('--border', fallbackGrid);
    const tickColor = getCssVariableValue('--muted-foreground', fallbackTick);
    const titleColor = getCssVariableValue('--foreground', fallbackTitle);
    const tooltipBgColor = getCssVariableValue('--popover', fallbackTooltipBg); // Or --card if preferred
    const tooltipTitleColor = getCssVariableValue('--popover-foreground', fallbackTooltipFore);
    const tooltipBodyColor = getCssVariableValue('--popover-foreground', fallbackTooltipFore);

    console.log(`[ChartTheme] Colors - Grid: ${gridColor}, Tick: ${tickColor}, Title: ${titleColor}, TooltipBg: ${tooltipBgColor}`);

    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { // Smoother interaction
            mode: 'index', // Show tooltips for all datasets at that index
            intersect: false,
        },
        plugins: {
            title: { display: false }, // Keep titles outside chart generally
            tooltip: {
                // mode: 'index', // Set in interaction now
                // intersect: false, // Set in interaction now
                backgroundColor: tooltipBgColor,
                titleColor: tooltipTitleColor,
                bodyColor: tooltipBodyColor,
                borderColor: gridColor,
                borderWidth: 1,
                titleFont: { weight: '500', family: 'Inter, sans-serif', size: 12 }, // Adjusted weight
                bodyFont: { family: 'Inter, sans-serif', size: 11 },
                padding: 8, // Slightly less padding
                boxPadding: 4,
                cornerRadius: 4, // Matches default Shadcn radius 'sm'
                usePointStyle: true, // Use point style in tooltip legend
                callbacks: {
                    // Callback to format the tooltip label more informatively
                    label: function(context) {
                        let label = context.dataset.label || '';
                        let value = context.parsed.y;

                        // Try to extract meaningful base label if formatted by utils.py
                        // E.g., "KPI_Name (Metric) - Vendor" -> "KPI_Name (Metric)"
                        let baseLabelMatch = label.match(/^(.*?)\s*(-.*Vendor.*)?$/);
                        let baseLabel = baseLabelMatch ? baseLabelMatch[1].trim() : label;
                        // Attempt to extract vendor if not already displayed (or customize as needed)
                        let vendorPart = label.includes('- V') ? '' : (context.dataset.vendor_name ? ` - ${context.dataset.vendor_name}` : '');

                        let displayValue = 'N/A';
                        if (value !== null && typeof value !== 'undefined' && !isNaN(value)) {
                             // Formatting based on inferred type (or dataset property)
                             const isPercentage = context.dataset?.metric?.includes('%') || baseLabel.includes('%');
                             if (isPercentage) {
                                 displayValue = value.toFixed(2) + '%';
                             } else if (Math.abs(value) >= 1000) {
                                // Simple large number formatting (e.g., 1.2k, 1.5M)
                                 displayValue = Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
                             } else {
                                displayValue = value.toFixed(2); // Default to 2 decimal places
                             }
                        }
                        return `${baseLabel}${vendorPart}: ${displayValue}`;
                    },
                    // Optional: Title callback if you want to format the date/time axis value
                    // title: function(tooltipItems) {
                    //     // Example: return tooltipItems[0].label; // Adjust formatting if needed
                    //     return tooltipItems[0].label;
                    // }
                }
            },
            legend: {
                display: true,
                position: 'bottom',
                align: 'center',
                labels: {
                    boxWidth: 8, // Smaller box
                    boxHeight: 8, // Match width
                    padding: 15,
                    font: { family: 'Inter, sans-serif', size: 11 },
                    color: tickColor, // Use muted color for legend text
                    usePointStyle: true, // Use point style in legend
                }
            }
        },
        elements: {
            point: {
                radius: 0, // Hide points by default
                hoverRadius: 4,
                hitRadius: 10,
                pointStyle: 'circle',
                backgroundColor: 'white', // Ensure hover point is visible
                borderWidth: 1 // Border on hover
            },
            line: {
                borderWidth: 2, // Slightly thicker lines
                tension: 0.3, // Smoother curves
                spanGaps: true // Connect points across null values
            }
        },
        layout: {
            padding: { top: 5, left: 0, right: 5, bottom: 0 }
        },
        scales: {
            x: {
                grid: {
                    display: false, // Hide vertical grid lines for cleaner look
                    // color: gridColor,
                    // drawTicks: false,
                    // drawBorder: false,
                },
                border: { // Show X axis line subtlely
                    display: true,
                    color: gridColor
                },
                ticks: {
                    maxRotation: 0, // Keep horizontal
                    minRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10, // Limit number of ticks
                    font: { family: 'Inter, sans-serif', size: 10 },
                    color: tickColor,
                    padding: 10 // More padding from axis
                }
            },
            y: { // Default Y axis (can be overridden)
                border: { // Hide Y axis line itself by default, grid is enough
                    display: false
                },
                grid: {
                    color: gridColor, // Main grid color
                    drawTicks: false, // Don't draw ticks on the grid lines
                    drawBorder: false, // Ensure axis line isn't drawn *by grid*
                },
                beginAtZero: false, // Let Chart.js decide based on data
                grace: '5%', // Add 5% padding above/below data range
                ticks: {
                    font: { family: 'Inter, sans-serif', size: 10 },
                    color: tickColor,
                    padding: 10, // More padding
                    precision: 2 // Default precision
                    // Callback for smarter formatting (e.g., percentages, large numbers)
                    // callback: function(value, index, ticks) {
                    //      // Example: Add % if axis ID indicates it
                    //      if (this.axis.id.includes('percent')) return value.toFixed(1) + '%';
                    //      return value.toLocaleString(); // Basic localization
                    // }
                }
                // Y Axis title configuration example (add if needed, generally better outside chart)
                // title: {
                //    display: true,
                //    text: 'Value',
                //    font: { size: 11, weight: '500', family: 'Inter, sans-serif'},
                //    color: tickColor, // Use tick color for less emphasis
                //    padding: { top: 0, bottom: 5 }
                // }
            }
        }
    };
}


/**
 * Provides a curated color palette inspired by Tailwind/Shadcn defaults.
 * Designed for better visual harmony than the default Chart.js colors.
 * @returns {string[]} An array of HSL color strings.
 */
function getShadcnChartColors() {
    // Colors carefully chosen to work well on both light/dark Shadcn backgrounds
    // Using HSL values similar to Tailwind defaults (e.g., blue-500, sky-500, red-500, etc.)
    return [
        'hsl(221, 83%, 53%)',   // Blue-600 (Primary-ish)
        'hsl(21, 90%, 57%)',   // Orange-600
        'hsl(134, 46.20%, 45.90%)',  // Green-600
        'hsl(347, 87%, 56%)',  // Red-600
        'hsl(262, 82%, 58%)',  // Violet-600
        'hsl(52, 96%, 53%)',   // Yellow-500
        'hsl(199, 98%, 48%)',  // Sky-500
        'hsl(320, 78%, 57%)',  // Pink-600
        'hsl(217, 33%, 50%)',  // Slate-600 (Muted)
        'hsl(84, 81%, 47%)',   // Lime-600
        'hsl(180, 86%, 41%)',  // Teal-600
        'hsl(281, 79%, 51%)'   // Purple-700
    ];
}

console.log("[ChartTheme] Shadcn chart theme helpers loaded.");