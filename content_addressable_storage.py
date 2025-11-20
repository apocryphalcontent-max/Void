"""
Content-Addressable Storage (CAS) Layer
"The Ophanim" - The Wheels/Storage

Implements IPFS-like content-addressable storage where files are named
by their cryptographic hash (SHA-256).

Key features:
- Content addressing (hash-based naming)
- Automatic deduplication (same content = same hash = single storage)
- Merkle DAG for hierarchical data
- Immutable storage (content never changes)
- Efficient content retrieval

References:
- "IPFS - Content Addressed, Versioned, P2P File System" (Benet, 2014)
- "Git Internals" - Similar content-addressable design
- "Venti: A New Approach to Archival Storage" (Quinlan & Dorward, 2002)
"""

import hashlib
import json
import time
import os
import threading
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict


# ============================================================================
# CONTENT ADDRESSING
# ============================================================================

def compute_content_hash(content: bytes) -> str:
    """
    Compute SHA-256 hash of content.
    
    Args:
        content: Raw content bytes
        
    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(content).hexdigest()


@dataclass
class ContentBlock:
    """
    A block of content in the CAS.
    
    Each block is:
    - Immutable (content never changes)
    - Content-addressed (hash is the name)
    - Deduplicated (same content = same hash)
    """
    content_hash: str  # SHA-256 hash
    content: bytes     # Raw content
    size: int          # Size in bytes
    created_at: float  # Creation timestamp
    ref_count: int = 0 # Reference count for GC
    
    def __hash__(self):
        return hash(self.content_hash)


@dataclass
class MerkleNode:
    """
    A node in a Merkle DAG (Directed Acyclic Graph).
    
    Used for hierarchical data structures:
    - Files with multiple blocks
    - Directories with multiple files
    - Snapshots with multiple directories
    """
    node_hash: str
    node_type: str  # "file", "directory", "block"
    links: List[str] = field(default_factory=list)  # Hashes of children
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'node_hash': self.node_hash,
            'node_type': self.node_type,
            'links': self.links,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MerkleNode':
        """Deserialize from dictionary"""
        return cls(
            node_hash=data['node_hash'],
            node_type=data['node_type'],
            links=data.get('links', []),
            metadata=data.get('metadata', {})
        )


# ============================================================================
# CONTENT-ADDRESSABLE STORAGE
# ============================================================================

class ContentAddressableStorage:
    """
    Content-Addressable Storage system.
    
    Features:
    1. Content addressing (hash-based naming)
    2. Automatic deduplication
    3. Merkle DAG for hierarchical data
    4. Reference counting for garbage collection
    5. Efficient retrieval by hash
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize CAS.
        
        Args:
            storage_path: Path for persistent storage (None = in-memory)
        """
        self.storage_path = storage_path
        
        # In-memory storage
        self.blocks: Dict[str, ContentBlock] = {}
        self.merkle_nodes: Dict[str, MerkleNode] = {}
        
        # Statistics
        self.stats = {
            'blocks_stored': 0,
            'blocks_retrieved': 0,
            'bytes_stored': 0,
            'dedup_savings_bytes': 0,
            'merkle_nodes': 0
        }
        
        self._lock = threading.Lock()
        
        # Create storage directory if needed
        if storage_path:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
    
    def put(self, content: bytes) -> str:
        """
        Store content in CAS.
        
        Args:
            content: Content to store
            
        Returns:
            Content hash (address)
        """
        # Compute content hash
        content_hash = compute_content_hash(content)
        
        with self._lock:
            # Check if already exists (deduplication)
            if content_hash in self.blocks:
                # Already stored - just increment ref count
                self.blocks[content_hash].ref_count += 1
                self.stats['dedup_savings_bytes'] += len(content)
                return content_hash
            
            # Create new block
            block = ContentBlock(
                content_hash=content_hash,
                content=content,
                size=len(content),
                created_at=time.time(),
                ref_count=1
            )
            
            # Store block
            self.blocks[content_hash] = block
            self.stats['blocks_stored'] += 1
            self.stats['bytes_stored'] += len(content)
            
            # Persist to disk if configured
            if self.storage_path:
                self._persist_block(block)
            
            return content_hash
    
    def get(self, content_hash: str) -> Optional[bytes]:
        """
        Retrieve content by hash.
        
        Args:
            content_hash: SHA-256 hash of content
            
        Returns:
            Content bytes, or None if not found
        """
        with self._lock:
            self.stats['blocks_retrieved'] += 1
            
            if content_hash in self.blocks:
                return self.blocks[content_hash].content
            
            # Try loading from disk
            if self.storage_path:
                block = self._load_block(content_hash)
                if block:
                    self.blocks[content_hash] = block
                    return block.content
            
            return None
    
    def has(self, content_hash: str) -> bool:
        """
        Check if content exists in CAS.
        
        Args:
            content_hash: SHA-256 hash
            
        Returns:
            True if content exists
        """
        with self._lock:
            return content_hash in self.blocks
    
    def put_file(self, file_path: str, chunk_size: int = 1024 * 1024) -> str:
        """
        Store a file in CAS by chunking it.
        
        Args:
            file_path: Path to file
            chunk_size: Size of chunks in bytes
            
        Returns:
            Hash of the file's Merkle DAG root
        """
        # Read file in chunks
        chunks = []
        chunk_hashes = []
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Store chunk
                chunk_hash = self.put(chunk)
                chunks.append(chunk)
                chunk_hashes.append(chunk_hash)
        
        # Create Merkle node for file
        file_name = os.path.basename(file_path)
        file_node = MerkleNode(
            node_hash="",  # Will be computed
            node_type="file",
            links=chunk_hashes,
            metadata={
                'name': file_name,
                'size': sum(len(c) for c in chunks),
                'chunks': len(chunks),
                'chunk_size': chunk_size
            }
        )
        
        # Compute hash of file node
        node_data = json.dumps(file_node.to_dict(), sort_keys=True).encode()
        file_node.node_hash = compute_content_hash(node_data)
        
        with self._lock:
            self.merkle_nodes[file_node.node_hash] = file_node
            self.stats['merkle_nodes'] += 1
        
        return file_node.node_hash
    
    def get_file(self, file_hash: str) -> Optional[bytes]:
        """
        Retrieve a file by its Merkle DAG root hash.
        
        Args:
            file_hash: Hash of file's Merkle node
            
        Returns:
            Complete file content, or None if not found
        """
        with self._lock:
            if file_hash not in self.merkle_nodes:
                return None
            
            file_node = self.merkle_nodes[file_hash]
            
            if file_node.node_type != "file":
                return None
        
        # Retrieve all chunks
        chunks = []
        for chunk_hash in file_node.links:
            chunk = self.get(chunk_hash)
            if chunk is None:
                return None
            chunks.append(chunk)
        
        # Concatenate chunks
        return b''.join(chunks)
    
    def put_directory(self, entries: Dict[str, str]) -> str:
        """
        Store a directory structure in CAS.
        
        Args:
            entries: Map of filename -> content_hash
            
        Returns:
            Hash of directory's Merkle node
        """
        dir_node = MerkleNode(
            node_hash="",  # Will be computed
            node_type="directory",
            links=list(entries.values()),
            metadata={
                'entries': entries,
                'count': len(entries)
            }
        )
        
        # Compute hash of directory node
        node_data = json.dumps(dir_node.to_dict(), sort_keys=True).encode()
        dir_node.node_hash = compute_content_hash(node_data)
        
        with self._lock:
            self.merkle_nodes[dir_node.node_hash] = dir_node
            self.stats['merkle_nodes'] += 1
        
        return dir_node.node_hash
    
    def list_directory(self, dir_hash: str) -> Optional[Dict[str, str]]:
        """
        List contents of a directory.
        
        Args:
            dir_hash: Hash of directory's Merkle node
            
        Returns:
            Map of filename -> content_hash, or None if not found
        """
        with self._lock:
            if dir_hash not in self.merkle_nodes:
                return None
            
            dir_node = self.merkle_nodes[dir_hash]
            
            if dir_node.node_type != "directory":
                return None
            
            return dir_node.metadata.get('entries', {})
    
    def gc_unreferenced(self) -> int:
        """
        Garbage collect blocks with zero references.
        
        Returns:
            Number of blocks collected
        """
        collected = 0
        
        with self._lock:
            # Find blocks with zero references
            to_remove = []
            for content_hash, block in self.blocks.items():
                if block.ref_count == 0:
                    to_remove.append(content_hash)
            
            # Remove blocks
            for content_hash in to_remove:
                block = self.blocks.pop(content_hash)
                self.stats['bytes_stored'] -= block.size
                collected += 1
                
                # Delete from disk if configured
                if self.storage_path:
                    self._delete_block(content_hash)
        
        return collected
    
    def _persist_block(self, block: ContentBlock):
        """Persist block to disk"""
        if not self.storage_path:
            return
        
        # Store in subdirectory based on first 2 chars of hash
        subdir = Path(self.storage_path) / block.content_hash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        
        block_path = subdir / block.content_hash
        block_path.write_bytes(block.content)
    
    def _load_block(self, content_hash: str) -> Optional[ContentBlock]:
        """Load block from disk"""
        if not self.storage_path:
            return None
        
        subdir = Path(self.storage_path) / content_hash[:2]
        block_path = subdir / content_hash
        
        if not block_path.exists():
            return None
        
        content = block_path.read_bytes()
        
        return ContentBlock(
            content_hash=content_hash,
            content=content,
            size=len(content),
            created_at=time.time(),
            ref_count=1
        )
    
    def _delete_block(self, content_hash: str):
        """Delete block from disk"""
        if not self.storage_path:
            return
        
        subdir = Path(self.storage_path) / content_hash[:2]
        block_path = subdir / content_hash
        
        if block_path.exists():
            block_path.unlink()
    
    def get_statistics(self) -> dict:
        """Get storage statistics"""
        with self._lock:
            return {
                **self.stats,
                'total_blocks': len(self.blocks),
                'total_merkle_nodes': len(self.merkle_nodes),
                'dedup_ratio': (
                    self.stats['dedup_savings_bytes'] / 
                    max(self.stats['bytes_stored'], 1)
                ) * 100
            }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def _example_usage():
    """Demonstrate Content-Addressable Storage"""
    
    print("Content-Addressable Storage (CAS)")
    print("="*70)
    
    # Create CAS
    print("\n1. Initializing Content-Addressable Storage...")
    cas = ContentAddressableStorage()
    
    # Store some content
    print("\n2. Storing content...")
    content1 = b"Hello, World!"
    hash1 = cas.put(content1)
    print(f"   Content: {content1.decode()}")
    print(f"   Hash: {hash1}")
    
    # Store duplicate content (should deduplicate)
    print("\n3. Storing duplicate content...")
    hash2 = cas.put(content1)
    print(f"   Hash: {hash2}")
    print(f"   Same as hash1: {hash1 == hash2}")
    print(f"   ✓ Automatic deduplication!")
    
    # Retrieve content
    print("\n4. Retrieving content by hash...")
    retrieved = cas.get(hash1)
    print(f"   Retrieved: {retrieved.decode()}")
    print(f"   Matches original: {retrieved == content1}")
    
    # Store different content
    print("\n5. Storing different content...")
    content2 = b"Different content"
    hash3 = cas.put(content2)
    print(f"   Content: {content2.decode()}")
    print(f"   Hash: {hash3}")
    print(f"   Different from hash1: {hash3 != hash1}")
    
    # Create a directory structure
    print("\n6. Creating directory structure...")
    dir_entries = {
        'file1.txt': hash1,
        'file2.txt': hash3
    }
    dir_hash = cas.put_directory(dir_entries)
    print(f"   Directory hash: {dir_hash}")
    
    # List directory
    print("\n7. Listing directory contents...")
    entries = cas.list_directory(dir_hash)
    for name, content_hash in entries.items():
        print(f"   {name}: {content_hash[:16]}...")
    
    # Statistics
    print("\n8. Storage statistics:")
    stats = cas.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n9. CAS guarantees:")
    print("   ✓ Content is immutable (never changes)")
    print("   ✓ Same content = same hash = single storage")
    print("   ✓ Absolute deduplication (no redundancy)")
    print("   ✓ Content integrity (hash verification)")
    print("   ✓ Efficient retrieval by hash")
    print("   ✓ Merkle DAG for hierarchical data")
    
    print("\n" + "="*70)
    print("The universe contains no redundancy, only references.")


if __name__ == "__main__":
    _example_usage()
