// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.html',   // Scans all .html files in the templates folder and subfolders
    './static/js/**/*.js',     // Scans all .js files in static/js (if you add/remove classes with JS)
    './forms.py',              // If you dynamically add classes in Python forms (less common for direct Tailwind classes)
    // Add any other paths where you use Tailwind classes
  ],
  darkMode: 'class', // To enable dark mode based on a class on the <html> element
  theme: {
    extend: {
      container: { // Optional: If you've customized the container
        center: true,
        padding: {
          DEFAULT: '1rem',
          sm: '2rem',
          lg: '4rem',
          xl: '5rem',
        },
      },
      colors: { // Your theme's custom colors from shadcn/ui (example)
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: { // Example border radius from shadcn/ui
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      // ... any other theme extensions
    },
  },
  plugins: [
    require('@tailwindcss/forms'), // If you are using this plugin
    // require('@tailwindcss/typography'), // If you are using this for `prose`
  ],
}