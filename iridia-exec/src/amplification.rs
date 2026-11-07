//! An amplification program on the CPU: a thread group that launches mesh thread groups. [`AmplificationShader::launches`]
//! runs the thread groups of a dispatch one after another and returns the launch each made: the mesh thread groups and
//! the words of the task payload it handed them. [`AmplificationShader::draw`] runs each launch's mesh thread groups too
//! ([`MeshShader::draw_launch`]) and returns their meshlets.
//!
//! The program is an amplification program on the IR ([`AmplificationShader::from_ir`]): the group's invocations run in
//! lockstep like a compute program's. Its `DispatchMesh` is a barrier every invocation reaches once (D3D requires it, and
//! a group where one does not is an error of the run), and the thread groups and payload that count are the group's
//! first invocation's.

use super::mesh::{MeshShader, Meshlet};
use super::stage::{Bindings, Constants};
use super::{Executable, error};
use crate::{Error, Sampler};
use iridia::{AmplificationFacts, Interface, StageFacts};

/// What one thread group of an amplification program launched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    /// The amplification thread group's position in its dispatch.
    pub group: [u32; 3],
    /// The mesh thread groups it launched, along x, y and z.
    pub groups: [u32; 3],
    /// The words of the task payload each of them reads.
    pub payload: Vec<u32>,
}

/// An amplification program ready to run.
#[derive(Debug)]
pub struct AmplificationShader {
    interface: Interface,
    executable: Executable,
    facts: AmplificationFacts,
    bindings: Bindings,
}

impl AmplificationShader {
    /// Prepare `program`, an amplification program on the IR, to run.
    pub fn from_ir(program: &iridia_ir::Program) -> Result<Self, Error> {
        let StageFacts::Amplification(facts) = program.interface.facts else {
            return Err(error(format!(
                "a {:?} program is not an amplification program",
                program.interface.stage
            )));
        };
        let executable = Executable::from_ir(program)?;
        Ok(Self {
            bindings: Bindings::new(&program.interface).map_err(error)?,
            interface: program.interface.clone(),
            executable,
            facts,
        })
    }

    /// The interface of the amplification program the shader runs.
    pub fn interface(&self) -> &Interface {
        &self.interface
    }

    /// What the program states: the bytes of its task payload.
    pub fn facts(&self) -> AmplificationFacts {
        self.facts
    }

    /// The thread group's invocations along each axis.
    pub fn group_size(&self) -> [u32; 3] {
        self.executable.workgroup.unwrap_or([1, 1, 1])
    }

    /// Run `groups` thread groups (`wave` the wave width the wave operations report) and return the launch each made,
    /// in the order of the dispatch (x, then y, then z). `constants` holds the rows of the constant buffers by
    /// (register, space); image and buffer access go to `sampler` by bind slot.
    pub fn launches(
        &self,
        constants: &Constants,
        sampler: &mut dyn Sampler,
        groups: [u32; 3],
        wave: u32,
    ) -> Result<Vec<Launch>, Error> {
        let bound = self.bindings.bind(&self.executable, constants).map_err(error)?;
        let mut backend = self.bindings.backend(sampler);
        let mut launches = Vec::new();
        self.executable
            .dispatch_amplification_elements(
                &bound,
                &mut backend,
                &mut Default::default(),
                &mut Default::default(),
                groups,
                wave,
                &mut |group, words| {
                    launches.push(self.read(group, words)?);
                    Ok(())
                },
            )
            .map_err(error)?;
        Ok(launches)
    }

    /// Run `groups` thread groups, then the mesh thread groups each launched with `mesh`, and return their meshlets:
    /// each launch's in the order of its dispatch, the launches in the order of theirs. The two programs' payloads
    /// must be the same size (Direct3D's runtime requires it); both read `constants` and `sampler`.
    pub fn draw(
        &self,
        mesh: &MeshShader,
        constants: &Constants,
        sampler: &mut dyn Sampler,
        groups: [u32; 3],
        wave: u32,
    ) -> Result<Vec<Meshlet>, Error> {
        if mesh.facts().payload != self.facts.payload {
            return Err(error(format!(
                "the amplification program's payload is {} bytes, the mesh program's {}",
                self.facts.payload,
                mesh.facts().payload
            )));
        }
        let mut meshlets = Vec::new();
        for launch in self.launches(constants, sampler, groups, wave)? {
            meshlets.extend(mesh.draw_launch(constants, sampler, &launch, wave)?);
        }
        Ok(meshlets)
    }

    /// The launch in the words a finished group left: how many invocations reached `DispatchMesh`, the thread groups,
    /// the payload's words.
    fn read(&self, group: [u32; 3], words: &[u32]) -> Result<Launch, String> {
        let [x, y, z] = self.group_size();
        let invocations = u64::from(x)
            .saturating_mul(u64::from(y))
            .saturating_mul(u64::from(z));
        let (Some(&reached), Some(Ok(groups)), Some(payload)) = (
            words.first(),
            words.get(1..4).map(<[u32; 3]>::try_from),
            words.get(4..),
        ) else {
            return Err(format!(
                "{} words of a launch, fewer than its counts",
                words.len()
            ));
        };
        if u64::from(reached) != invocations {
            return Err(format!(
                "{reached} of the group's {invocations} invocations reached DispatchMesh, which every one reaches once"
            ));
        }
        Ok(Launch {
            group,
            groups,
            payload: payload.to_vec(),
        })
    }
}
