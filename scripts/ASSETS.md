# Large assets (.blend / .ply)

Large 3D files are **not stored in git**. The git tree only holds code and small
files; the big binaries live in a **public Backblaze B2 bucket** and are synced
with [rclone](https://rclone.org) (push) / `curl` (pull).

This replaced Git LFS, which can't hold files over 2 GB on GitHub (e.g.
`cave-maps.blend` is ~4.4 GB).

Assets are organised into subdirectories under `src/assets/`:

- `point_clouds/` — `.ply` point clouds (chamber, tunnels, grapevine, combined, …)
- `blender/` — `.blend` scene files (`cave-maps.blend`, `stanford.blend`)

The manifest records these paths relative to `src/assets/`, and pulls recreate
the same layout.

## Fetching assets (no account needed)

The bucket is public-read, so pulling requires no credentials:

```bash
./scripts/assets-pull.sh
```

It downloads everything listed in `assets-manifest.txt`. `curl -C -` resumes
interrupted transfers, so just re-run if a big file drops.

> If it 404s, check the `ENDPOINT` at the top of `assets-pull.sh` matches the
> "Endpoint" shown on the bucket's details page in Backblaze.

## Uploading assets (maintainers)

```bash
# one-time, if rclone is missing:  https://rclone.org/downloads/
./scripts/assets-push.sh
```

The `keyID` is hardcoded (not sensitive). The script **prompts** only for the
secret `applicationKey` each run and uses it in-memory only — **nothing is
written to disk or committed**. Get a key from Backblaze → **App Keys**.
After a push it refreshes `assets-manifest.txt`; commit that file.

Only `*.blend` and `*.ply` sync (see `assets-filter.txt`); `.blend1` autosave
backups are excluded. Push uses `rclone copy` and never deletes remote files.

## Config

| What | Where | Default |
|------|-------|---------|
| Bucket name | `PROSPECTOR_ASSETS_BUCKET` env, or script default | `isaacronaldward-prospector` |
| S3 endpoint (pull) | `PROSPECTOR_ASSETS_ENDPOINT` env, or `assets-pull.sh` | `s3.us-west-004.backblazeb2.com` |
| B2 credentials (push) | prompted at run time | — never stored |
