# QMCPy Shared Leadership Roles and Responsibilities

This document outlines the leadership structure for QMCPy open-source software development and the responsibilities associated with each lead role. These positions are designed to distribute oversight across key areas of the project while fostering collaboration and sustainable growth.

## Leadership Structure

### Executive Committee
Sou-Cheng T. Choi, Fred Hickernell, Aleksei Sorokin

* Provide strategic direction for QMCPy,
* Coordinate between all leadership roles,
* Ensure alignment with the project's long-term vision and community goals.

---

## Core Lead Roles

### 1. Theory
**Lead:** Fred Hickernell

#### Key Responsibilities
- **Mathematical Foundations**: Ensure the theoretical correctness and rigor of QMCPy algorithms and implementations.
- **Algorithm Review**: Review PRs involving new or modified algorithms for mathematical soundness.
- **Research Alignment**: Connect QMCPy development with current advances in quasi-Monte Carlo theory and related fields.
- **Collaboration with Authors**: Facilitate communication between software contributors and researchers publishing results that use or extend QMCPy.
- **Educational Content**: Guide the creation of documentation and demos that accurately convey the mathematical ideas behind QMCPy methods.

#### Strategic Goals
- Maintain QMCPy's standing as a mathematically rigorous and research-grade software package.
- Bridge the gap between theoretical research and practical implementation.
- Foster contributions from the QMC research community.

---

### 2. Release
**Lead:** Aleksei Sorokin  

**Co-Lead:** Richard Varela

#### Key Responsibilities
- **Pre-Release Validation**: Ensure all components are ready for release by verifying that documentation compiles correctly and all tests pass.
- **Publishing to PyPI**: Execute the publication process to make QMCPy available via `pip install qmcpy`.
- **GitHub Releases**: Create and manage official GitHub releases with appropriate version tagging and release notes.
- **Quality Assurance**: Serve as the last line of defense against broken installations, which are critical to maintaining user trust and satisfaction.
- **Coordination**: Work closely with Documentation, Test, and Blog leads to ensure all aspects are polished before release.
- **Timeline Management**: Pull requests into `develop` may occur at any time, but push into `master` is an occasional event.  Coordinate release schedules and communicate timelines to the team (e.g., version 2.1 target: December 10).

#### Strategic Goals
- Ensure smooth and reliable releases of QMCPy.
- Maintain high-quality standards for all releases.
- Foster collaboration between leads to ensure readiness for each release.


---

### 3. Documentation and Communication
**Lead:** Sou-Cheng Choi

**Co-Lead:** Jiangrui Kang and Larysa Matiukha

#### Key Responsibilities
- **Documentation Standards**: Ensure all pull requests (PRs) with new components include appropriate documentation for those components.
- **Demo Integration**: Verify that demos are properly rendered into the documentation website.
- **API Documentation**: Maintain comprehensive package reference documentation.
- **Content Development**: Oversee the creation of blog posts, tutorials, and other content that highlights QMCPy's features, research applications, and community contributions.
- **Editorial Oversight**: Coordinate peer reviews for blog posts and other content, ensuring quality and alignment with QMCPy's goals.
- **Website Management**: Oversee the "Documentation" and "Blogs" sections of the QMCPy website, ensuring they are up-to-date and well-integrated with other sections like "News" and "Events."
- **Content Strategy**: Develop a cohesive strategy for documentation and communication topics, aligning them with QMCPy's broader goals.
- **Collaboration**: Work closely with other leads to ensure documentation and communication efforts complement the overall project.
- **Community Engagement**: Use blogs, tutorials, and other content to showcase user stories, case studies, and research applications, fostering a sense of community.

#### Strategic Goals
- Ensure QMCPy's documentation is comprehensive, accessible, and user-friendly for both developers and practitioners.
- Highlight QMCPy's capabilities and community contributions through engaging content.
- Strengthen the connection between the "Documentation" and "Blogs" sections and other website areas, such as "News" and "Events."
- Encourage team-wide participation in content creation to maintain a diverse and active presence.
- Consider merging `qmcpy.org` with the documentation.

---

### 4. Test
**Lead:** Sou-Cheng Choi

**Co-Lead:** Brandon Sharp (probationary)

#### Key Responsibilities
- **Test Coverage**: Ensure all PRs with new components include both doctests and unit tests where applicable.
- **PR Validation**: Verify that all tests pass before pull requests are merged into the develop branch.
- **Demo Testing**: Ensure that demos included in PRs can be run successfully.
- **Test Automation**: Work toward automating the testing of demos and notebooks (emerging challenge as the project evolves).
- **Continuous Integration**: Maintain and improve CI/CD workflows for automated testing.
- **Test Documentation**: Document testing procedures and best practices for contributors.
- **Quality Standards**: Set and enforce testing standards across the codebase.

#### Strategic Goals
- Develop robust automated testing frameworks for Jupyter notebooks and demos.
- Maintain high test coverage as the package grows.
- Balance comprehensive testing with development velocity.

---

### 5. Impact

**Lead:** TBD

#### Key Responsibilities
- **Visual Identity**: Lead the design and implementation of a professional QMCPy logo incorporating the package name.
- **Repository Management**: Strategically rename repositories (e.g., QMCSoftware/QMCSoftware → QMCSoftware/qmcpy) while maintaining backward compatibility.
- **Professional Presentation**: Implement elements found in successful scientific software repositories:
  - Code of Conduct
  - Badges (tests, docs, PyPI, DOI, etc.)
  - Enhanced README formatting
  - Contributing guidelines
- **Distribution Channels**: Oversee publishing QMCPy to additional distribution platforms (e.g., Conda) beyond PyPI.
- **Best Practices**: Research and implement development and community engagement practices from popular scientific packages (e.g., NumPy, SciPy, Pandas).
- **Community Building**: Explore opportunities to integrate QMCPy into other packages and ecosystems to increase visibility and sustainability.
- **Outreach**: Support efforts to promote QMCPy at conferences, workshops, and through publications.

#### Strategic Goals
- Enhance package visibility and professional appearance.
- Improve user experience and onboarding.
- Position QMCPy competitively within the scientific Python ecosystem.

---

## Cross-Cutting Principles and Expectations

### Time Commitment
- While individual tasks require only a few minutes, diligence and attention to detail are essential.
- Attend team meetings every two weeks and quarterly leadership meetings as scheduled.
- Releases typically occur a few times per year, with preparatory work distributed across the release cycle.

### Collaboration and Communication
- All leads should maintain open communication channels (e.g., Slack, WhatsApp group for leads, email).
- Regular coordination meetings to align on release timelines and priorities.
- Transparent decision-making and inclusive community engagement.

### Sustainability and Rotation
- Leadership roles are typically committed for one-year terms, with the possibility of renewal.
- Periodic rotation of responsibilities helps prevent burnout and develops a broader leadership base.
- Co-leads provide continuity and redundancy, facilitating knowledge transfer.

### Quality and Standards
- All leads share responsibility for maintaining high-quality standards in their respective areas.
- PRs should be reviewed by the relevant lead(s) before merging.
- Each lead serves as a subject matter expert and resource for contributors in their domain.

### Growth and Innovation
- Leads are encouraged to explore new tools, methodologies, and best practices in their areas.
- Balance innovation with stability to maintain user trust.
- Document processes and decisions to build institutional knowledge.

### “Great Expectations” for All Leads
1. **Oversight and Accountability**: Monitor activities in your area of responsibility and ensure standards are met.
2. **Mentorship**: Support contributors and co-leads, answering questions and providing guidance.
3. **Process Documentation**: Maintain clear documentation of workflows and procedures.
4. **Proactive Communication**: Identify issues early and communicate with other leads and the broader team.
5. **Community Building**: Foster an inclusive, welcoming environment for all contributors.
6. **Long-term Thinking**: Consider the sustainability and scalability of processes and decisions.




