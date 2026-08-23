# How to download and store lunar images (team guide)

Follow these steps exactly. Don't skip steps or rename things — the
scripts expect specific folder names.

---

## Step 0 — One-time setup (do this once)

1. Ask Piyush to share the Google Drive folder called
   `DepthWizard_Lunar_Data` with your Gmail address.
2. Go to google.com/drive/download and install **Google Drive for
   Desktop** on your computer.
3. Sign in with the Gmail that got the invite.
4. After it installs, you'll see a new drive/folder on your computer
   (usually shows up as `G:` on Windows, or under "Google Drive" in
   Finder on Mac). Open it and confirm you can see
   `DepthWizard_Lunar_Data`.
5. Right-click the `DepthWizard_Lunar_Data` folder inside Drive and make
   sure it says something like **"Available offline"** (not "Available
   online only"). If it says online only, click to change it. This makes
   sure the files are actually saved on your computer, not just visible
   when you have internet.

You only do this once. After this, the folder just works like a normal
folder on your computer.

---

## Step 1 — Download an image from PRADAN

1. Go to `chmapbrowse.issdc.gov.in` and log in (register first if you
   haven't).
2. Click **Step 1: Select Instrument** → check **OHRC**.
3. Click **Step 2: Select PDS Product Type** → check **Calibrated**.
4. Click **Step 3: Select Area of Interest** → type in a box like this:

   ```
   Min Lat: -70
   Max Lat: -65
   Min Lon: -2.5
   Max Lon: 2.5
   ```

   (The website will complain if your box is bigger than 5 degrees in
   any direction — keep it small like the example above. You can try
   different Lon numbers if you get no results, but keep Min Lat and
   Max Lat close to -60 to -75, not right at -90. Right at -90 gives
   bad images — either totally black or way too bright.)

5. Click **Search**.
6. You'll get a list of results. Click each **Product ID** (the text,
   not the checkbox) to preview it. Look for one that shows real moon
   surface with craters — not a plain black square, not a noisy striped
   square.

   **Important: the preview image looks worse than the real data.**
   Even if the preview looks kind of dark or ugly, download it anyway —
   we have a script that checks the real quality properly. Only skip
   ones that are 100% pure black or 100% static/noise.

7. Tick the checkbox next to the one you picked, click **Download**.

---

## Step 2 — Unzip it into the right place

1. Find the `.zip` file you just downloaded (usually in your Downloads
   folder).
2. Right-click it → **Extract All** (or use 7-Zip / WinRAR if you have
   it).
3. You'll get a folder. Look at the folder name — it'll be something
   like `ch2_ohr_ncp_20241115T1326321339_d_img_d18`. That whole folder
   name matters — don't rename it.
4. **Move that whole folder** into:

   ```
   DepthWizard_Lunar_Data / raw_downloads /
   ```

   So the final path looks like:

   ```
   DepthWizard_Lunar_Data/raw_downloads/ch2_ohr_ncp_20241115T1326321339/
   ```

5. Inside that folder you should see 4 subfolders: `data`, `browse`,
   `geometry`, `miscellaneous`. Leave them exactly as they are — don't
   move, rename, or delete anything inside.

6. Wait a minute or two for the little Google Drive icon (bottom right
   of your screen, near the clock) to stop spinning — that means it
   finished uploading to the shared folder so everyone else can see it
   too.

---

## Step 3 — Run one script to check it and log it

This step tells everyone (through one shared spreadsheet) whether the
image is good, and what its numbers are (how sharp, what sun angle,
etc). You need Python installed for this — same one used for the main
DepthWizard app.

1. Open a terminal (PowerShell on Windows, Terminal on Mac).

2. Copy-paste this, but replace `<product_id>` with your actual folder
   name from Step 2:

   ```powershell
   python "G:\My Drive\DepthWizard_Lunar_Data\scripts\catalog_ohrc.py" "G:\My Drive\DepthWizard_Lunar_Data\raw_downloads\<product_id>" --catalog "G:\My Drive\DepthWizard_Lunar_Data\catalog.csv" --previews "G:\My Drive\DepthWizard_Lunar_Data\previews"
   ```

   Example, filled in:

   ```powershell
   python "G:\My Drive\DepthWizard_Lunar_Data\scripts\catalog_ohrc.py" "G:\My Drive\DepthWizard_Lunar_Data\raw_downloads\ch2_ohr_ncp_20241115T1326321339" --catalog "G:\My Drive\DepthWizard_Lunar_Data\catalog.csv" --previews "G:\My Drive\DepthWizard_Lunar_Data\previews"
   ```

   (If your Drive doesn't show up as `G:`, look for wherever
   `DepthWizard_Lunar_Data` actually lives on your computer and use
   that path instead.)

3. It'll print something like:

   ```
   [catalog_ohrc] usability_flag: OK  |  mean=119.9  std=31.7
   [catalog_ohrc] sun_elevation_deg: 8.4  pixel_resolution_m: 0.24
   ```

   - If it says `usability_flag: OK` — good, you're done, move to the
     next image if you want to download more.
   - If it says `SUSPECT_MOSTLY_BLACK` or `SUSPECT_LOW_CONTRAST` — this
     image probably isn't usable. Open the preview PNG it made (in the
     `previews` folder) to double check, then try a different image
     from Step 1 if it really is bad.

That's it. You don't need to do anything else — Piyush will pick up
the image from the shared folder for the depth pipeline.

---

## Things to NOT do

- Don't rename any downloaded folder or file.
- Don't move `data`, `browse`, `geometry`, `miscellaneous` around inside
  a product folder.
- Don't manually edit `catalog.csv` — it's built automatically.
- Don't send zip files over WhatsApp/Discord/email — always put them in
  the shared Drive folder. They're too big and it just causes confusion
  about which version is the real one.
- Don't download from the extreme South Pole (around -85 to -90
  latitude) — those images are usually either pure black or way too
  bright to use.

---

## Quick checklist every time you download a new image

- [ ] Downloaded a **Calibrated** OHRC image, latitude roughly -60 to
      -75 (not right at the pole)
- [ ] Unzipped into `raw_downloads/<product_id>/` with folder name
      untouched
- [ ] Waited for Google Drive to finish syncing
- [ ] Ran `catalog_ohrc.py` on it
- [ ] Checked it says `usability_flag: OK`

If all 5 boxes are checked, you're done — the image is ready for the
rest of the team to use.