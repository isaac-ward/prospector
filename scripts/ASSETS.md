# Large assets (.blend / .ply)

Large 3D files are **not stored in git**. The git tree only holds code and small
files; the big binaries live on an external object store (Google Drive by
default) and are synced with [rclone](https://rclone.org).

This replaced Git LFS, which can't hold files over 2 GB on GitHub (e.g.
`darpa2.blend` is ~3.2 GB).

## One-time setup

```powershell
winget install Rclone.Rclone      # if you don't have rclone
rclone config                     # create a remote named 'gdrive' (type: drive)
```

During `rclone config`: choose `n` (new remote), name it `gdrive`, storage type
`drive` (Google Drive), accept the defaults, and complete the browser login.

Using a different backend (Cloudflare R2, Backblaze B2, S3, ...) or folder? Set
`PROSPECTOR_ASSETS_REMOTE`, e.g. `r2:prospector-assets`.

## Daily use

```powershell
./scripts/assets-pull.ps1     # after cloning, or to get the latest assets
./scripts/assets-push.ps1     # after editing/adding a .blend or .ply
```

Only `*.blend` and `*.ply` sync (see `assets-filter.txt`). `.blend1` autosave
backups are excluded. `push` uses `rclone copy` and never deletes remote files.
