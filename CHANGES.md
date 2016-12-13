0.2 (unreleased)
----------------

- Added a new potential class for the Satoh density (Satoh 1980).
- Added support for Leapfrog integration when generating mock
    stellar streams.
- Added new colormaps and defaults for the matplotlib style.
- Added support for non-inertial reference frames and implemented a
    constant rotating reference frame.
- Added a new class - `Hamiltonian` - for storing potentials with
    reference frames and should be used for orbit integration.
- Added a new mock stream argument to output orbits of all of the
    mock stream star particles to an HDF5 file.

0.1.1 (2016-05-20)
------------------

- Removed debug statement.
- Added 'Why' page to documentation.

0.1.0 (2016-05-19)
------------------

- Initial release.
