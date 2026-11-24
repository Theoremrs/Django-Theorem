# Versioned release builds

The **Versioned release builds** workflow builds Python distributions and a
complete source snapshot on GitHub-hosted Linux. It does not deploy to Fly.

Run it manually from Actions for a candidate build. The initial
`codex/versioned-release-builds` branch also runs it on pushes so the workflow
can be verified before it reaches the default branch. Each run retains its
wheel, source distribution, complete source archive, build manifest and
SHA-256 checksums as an Actions artifact for 30 days.

Push a `v*` or `build-*` tag at the intended commit to keep the successful
artifacts in a draft GitHub prerelease. Tag builds do not publish a stable
release or update an existing release. Review the draft and its build result
before publishing. To retry a published version, create a new version rather
than replacing its assets.

The wheel's version comes from `pyproject.toml`; a snapshot label or Git tag
does not rewrite package metadata. Change the project version before creating
a semantic package release. No package is uploaded to PyPI.

These distributions contain Python source, not Python itself, native libraries,
model weights or provisioned services. The complete source archive includes
the deployment recipes and commands omitted from a Python wheel. Existing
worker-container publish workflows remain separate. Checksums and the manifest
identify the built source; they do not constitute a live application test.

After downloading all assets into one directory, run `sha256sum --check SHA256SUMS`
(or `shasum -a 256 --check SHA256SUMS` on macOS).
