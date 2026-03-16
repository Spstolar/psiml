# PsiML Website

This is the repository for the PsiML website, built with [Astro](https://astro.build) and [Quarto](https://quarto.org).

## Local Development

To run the site locally and preview changes:

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```
   *(To stop the server at any time, press `Ctrl + C` in the terminal)*

3. **Preview the site:**
   Open your browser and navigate to [http://localhost:4321](http://localhost:4321).

### Stopping a Background Server

If you accidentally started the server in the background (or closed the terminal window it was running in), you won't be able to stop it with `Ctrl + C`. You can find and stop it manually using `lsof` (List Open Files).

`lsof` is a command-line utility that lists all open files and the processes that opened them. In Unix-like systems, everything is a file, including network connections (ports).

1. Find the process using port 4321 (the default Astro port):
   ```bash
   lsof -i :4321
   ```
2. Look for the `PID` (Process ID) in the output.
3. Kill the process by running:
   ```bash
   kill -9 <PID>
   ```

Alternatively, you can forcefully kill any running Astro dev server by its name:
```bash
pkill -f "astro dev"
```

## Quarto Integration

This project integrates Quarto for technical documentation and older posts. The Quarto project is located in `quarto-content/`.

* The Quarto site is built into the Astro `public/quarto` directory. Astro automatically serves anything in the `public/` directory as static assets.
* To link to Quarto pages from Astro, simply point to the `/quarto/` path. For example, to link to the Quarto index page from Astro:
  ```html
  <a href="/quarto/index.html">Older Posts</a>
  ```
