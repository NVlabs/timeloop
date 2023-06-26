#define BOOST_TEST_MODULE TestCompoundConfig

#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <compound-config/compound-config.hpp>

// Number of testing cycles to run.
int TESTS = 100000;
// The seed for the entropy source.
uint SEED = 42;
// Whether or not to print sub-test progress reports.
bool progressReports = (getenv("TIMELOOP_UNITTEST_REPORT_PROGRESS") != NULL) &&
                (strcmp(getenv("TIMELOOP_UNITTEST_REPORT_PROGRESS"), "0") != 0);
// Whether or not to print runtime states.
bool runtimeStates = (getenv("TIMELOOP_UNITTEST_REPORT_RUNTIME_STATES") != NULL) &&
              (strcmp(getenv("TIMELOOP_UNITTEST_REPORT_RUNTIME_STATES"), "0") != 0);
// Whether or not to print end-of-test object states.
bool endStateReport = (getenv("TIMELOOP_UNITTEST_REPORT_END_STATES") != NULL) &&
               (strcmp(getenv("TIMELOOP_UNITTEST_REPORT_END_STATES"), "0") != 0);

// Changes the max random value to the U_LONG32 random value.
#undef RAND_MAX
#define RAND_MAX ULONG_LONG_MAX

// The location of the test files.
std::string TEST_LOC = "./src/unit-tests/compound-config/tests/";

// Static YAML file names we want to load in for the test.
std::map<std::string, std::vector<std::string>> FILES = {
    // https://github.com/Accelergy-Project/timeloop-accelergy-exercises
    {
        "accelergy-project/2020.ispass/timeloop/00/", {
            "1level.arch.yaml",
            "conv1d-1level.map.yaml",
            "conv1d.prob.yaml",

            "timeloop-model.ART_summary.yaml",
            "timeloop-model.ART.yaml",
            "timeloop-model.ERT_summary.yaml",
            "timeloop-model.ERT.yaml",
            "timeloop-model.flattened_architecture.yaml",
        }
    },
    {
        "accelergy-project/2020.ispass/timeloop/01/", {
            "2level.arch.yaml",
            "conv1d-2level-os.map.yaml",
            "conv1d-2level-ws.map.yaml",
            "conv1d.prob.yaml",

            "os/timeloop-model.ART_summary.yaml",
            "os/timeloop-model.ART.yaml",
            "os/timeloop-model.ERT_summary.yaml",
            "os/timeloop-model.ERT.yaml",
            "os/timeloop-model.flattened_architecture.yaml",

            "ws/timeloop-model.ART_summary.yaml",
            "ws/timeloop-model.ART.yaml",
            "ws/timeloop-model.ERT_summary.yaml",
            "ws/timeloop-model.ERT.yaml",
            "ws/timeloop-model.flattened_architecture.yaml",
        }
    },
    {
        "accelergy-project/2020.ispass/timeloop/02/", {
            "2level.arch.yaml",
            "conv1d+oc-2level-os-tiled.map.yaml",
            "conv1d+oc-2level-os.map.yaml",
            "conv1d+oc.prob.yaml",

            "tiled/timeloop-model.ART_summary.yaml",
            "tiled/timeloop-model.ART.yaml",
            "tiled/timeloop-model.ERT_summary.yaml",
            "tiled/timeloop-model.ERT.yaml",
            "tiled/timeloop-model.flattened_architecture.yaml",

            "untiled/timeloop-model.ART_summary.yaml",
            "untiled/timeloop-model.ART.yaml",
            "untiled/timeloop-model.ERT_summary.yaml",
            "untiled/timeloop-model.ERT.yaml",
            "untiled/timeloop-model.flattened_architecture.yaml",
        }
    }
};

/**
 * testScalarLookup makes sure for a Map CompoundConfigNode that contains a
 * Scalar value mapped to a key, the contained Scalar value agrees with a 
 * corresponding reference YAML::Node. 
 * 
 * The return value defaults to false if there is a conversion error and does
 * not raise any errors in BOOST or the runtime environment. 
 * 
 * A BOOST error is raised if there is an equality error.
 * 
 * @tparam T    The type we wish the scalar to resolve to.
 * 
 * @param CNode The CompoundConfigNode (CCN) thats value is being tested for. It
 *              is expected that a Map CCN is passed in, as CCN's lookupValue
 *              function, which is how it accesses Scalars, only works for
 *              key-value pairs.
 * @param YNode The YAML::Node that serves as a source of truth/reference value
 *              for CNode. It is expected that a YAML::Map Node is passed in to
 *              parallel the CNode structure.
 * @param key   The key corresponding to the Scalar value we want to compare.
 * 
 * @return      Returns whether a value at a key is Scalar AND Equal.
 */ 
template <typename T>
bool testScalarLookup(  const config::CompoundConfigNode& CNode, 
                        const YAML::Node& YNode, const std::string& key)
{
    /* First attempt to run the test as normal, keeping an eye out for a YAML
     * conversion error from YAML types to C++ types. */
    try {

        // predeclares values
        T expectedScalar, actualScalar;
        // value resolution
        expectedScalar = YNode[key].as<T>();

        // if successful resolution by CNode
        if (CNode.lookupValue(key, actualScalar))
        {
            // check equality with BOOST so that it is logged in the tester.
            BOOST_CHECK_EQUAL(expectedScalar, actualScalar);
            // and propagate equality check
            return expectedScalar == actualScalar;

        // otherwise return false since no lookupValue resolution
        } else
        {
            return false;
        }

    /* if there is a conversion error for the type we're trying to access,
     * return false. */
    } catch(const YAML::TypedBadConversion<T>& e) {
        // defaults to false on bad conversion
        return false;
    }
}

// Forward declaration.
bool testMapLookup(const config::CompoundConfigNode& CNode, const YAML::Node&YNode);
/**
 * testSequenceLookup makes sure for a Sequence CompoundConfigNode, the elements
 * in the Sequence agree with a reference YAML::Node.
 * 
 * The return value defaults to true until an element inequality is found. BOOST
 * raises errors if the CCN is not a List or Array, or the YAML::Node is not a
 * Sequence.
 * 
 * There is a difference execution pathway depending on whether or not the CCN
 * is an Array or List, as CCN provides a lookupArrayValue function that needs
 * to be tested. There are BOOST errors in code that depend on the execution
 * pathway.
 * 
 * @param CNode The CompoundConfigNode whose elements are being equality tested.
 *              A List/Array (Sequence in YAML) CCN is expected to be passed in.
 * @param YNode The YAML::Node that is a Sequence (either a List/Array in CCN)
 *              which serves as a source of truth/reference the CCN elements are
 *              being compared against.
 * 
 * @return      Returns whether all the elements in CNode and YNode agree.
 *              Returns true until proven otherwise.
 */
bool testSequenceLookup(const config::CompoundConfigNode& CNode,
                        const YAML::Node& YNode)
{
    /* Return value namespace + initialization. It defaults to true until an
     * inequality is found. */
    bool equal = true;

    // Checks that the children are YAML Sequences or equivalent.
    BOOST_CHECK(CNode.isList() || CNode.isArray());
    BOOST_CHECK(YNode.IsSequence());

    /* If the YNode children are Scalar (if ANY of them are Scalar), it's the 
     * equivalent of an Array in CNode. Therefore we should run the Array test 
     * execution pathway which triggers the CCN testArrayLookup function. */ 
    if (YNode[0].IsScalar())
    {
        // Confirms the CNode is an Array. Raises a BOOST error if not.
        BOOST_CHECK(CNode.isArray());

        // The heap space allocated for the CCN to dump its Array values into.
        std::vector<std::string> actual;

        /* Fetches Array, should always work if our previous code works as 
         * intended (i.e. the assumption, which should be the standard, that if 
         * ANY child of the YNode is a Scalar, the CNode is an Array). If not,
         * we raise a BOOST error. */
        BOOST_CHECK(CNode.getArrayValue(actual));

        /* Iterates over the YNode elements and compares them to the 
         * corresponding fetched element. Update return value to false if an
         * inequality is found. */
        for (int i = 0; (size_t) i < YNode.size(); i++)
        {
            equal = equal && actual[i] == YNode[i].as<std::string>();
        }
    // Otherwise, run the List execution pathway, as we are guaranteed a List.
    } else
    {
        // Verify that the CNode is a List, raising a BOOST error if not.
        BOOST_CHECK(CNode.isList());

        // Then go through all elements in the Sequence YNode.
        for (int i = 0; (std::size_t) i < YNode.size(); i++)
        {
            // Unpacks elements.
            config::CompoundConfigNode childCNode = CNode[i];
            YAML::Node childYNode = YNode[i];

            // If we unpacked a nested Sequence, recurse to check equality.
            if (childYNode.IsSequence())
            {
                equal = equal && testSequenceLookup(childCNode, childYNode);
            // Otherwise, we expect a Sequence of labeled values (Map).
            } else
            {
                // Check it is a sequence of Maps, raising a BOOST error if not.
                BOOST_CHECK(childCNode.isMap());
                BOOST_CHECK(childYNode.IsMap());

                /* Checks equality in the child nodes via testMapLookup. This
                 * method only works because values are always labeled. */ 
                equal = equal && testMapLookup(childCNode, childYNode);
            }
        }
    }

    return equal;
}

// Forward declaration.
bool mapNodeEq( const config::CompoundConfigNode CNode, const YAML::Node YNode, 
                const std::string& key, const YAML::NodeType::value TYPE);
/**
 * testMapLookup makes sure for a Map CompoundConfigNode, the key-value pairs in 
 * the Map agree with a reference YAML::Node.
 * 
 * The return value defaults to true until an element inequality is found.
 * 
 * @param CNode The CompoundConfigNode whose key-value pairs are being equality
 *              tested. A Map CCN is expected to be passed in.
 * @param YNode The Map YAML::Node which serves as a source of truth/reference 
 *              the CCN key-value pairs are being compared against.
 * 
 * @return      Returns whether the key-value pairs in CNode and YNode agree.
 *              returns true until proven otherwise.
 */
bool testMapLookup(const config::CompoundConfigNode& CNode, const YAML::Node&YNode)
{
    // Defines the return value namespace and instantiates it as true.
    bool equal = true;

    // Iterates over all key-value pairs in YNode and compares to CNode.
    for (auto nodeMapPair:YNode)
    {
        // Extracts the key from YNode.
        const std::string key = nodeMapPair.first.as<std::string>();
        /* Checks the value at key are equal. Note that we did not unpack the
         * value at the key for either the CNode or the YNode. This is because
         * if the value at the key is a Scalar, we need to preserve the packing
         * given CompoundConfigNode can only access packed Scalars. However,
         * since a Map value can be anything, we need to abstract the logic
         * of testing for equality here to another function as the CCN expected
         * behavior is different depending on the value type. */
        equal = equal && mapNodeEq(CNode, YNode, key, nodeMapPair.second.Type());
    }

    return equal;
}

/** 
 * Checks if a value corresponding to a certain key in a given Map CCN and
 * corresponding reference Map YAML::Node.
 * 
 * @param CNode A Map CCN which is assumed to have a value corresponding to the
 *              key we're trying to access.
 * @param YNode The reference Map YAML::Node which is assumed to have have a
 *              value corresponding to the key we're trying to access.
 * @param key   The key we're trying to compare.
 * @param TYPE  The expected type of the value we're unpacking/checking the
 *              equality of.
 * 
 * @return      Returns whether the CNode is equal to the reference. Defaults to
 *              false.
 */
bool mapNodeEq( const config::CompoundConfigNode CNode, const YAML::Node YNode, 
                const std::string& key, const YAML::NodeType::value TYPE)
{
    // Declares the namespace of the return value. Instantiates to false.
    bool nodeEq = false;

    /* Declares the namespace of the child nodes we'd unpack if we're accessing
     * a Sequence or Map for ownership reasons. Switch statements don't like
     * declaring variables depending on which execution pathway is taken. */
    config::CompoundConfigNode childCNode;
    YAML::Node childYNode;

    // Determines what check to do based off child node type.
    switch(TYPE)
    {
        // Nulls have nothing, so there is no length. Scalars have length 1.
        case YAML::NodeType::Null:
            nodeEq = CNode.lookup(key).getLength() == 0;
            break;
        // Tests all possible scalar output values because there's 9 of them.
        case YAML::NodeType::Scalar:
            // Tests the integer types.
            nodeEq = testScalarLookup<double>(CNode, YNode, key) || nodeEq;
            nodeEq = testScalarLookup<bool>(CNode, YNode, key) || nodeEq;
            nodeEq = testScalarLookup<int>(CNode, YNode, key) || nodeEq;
            nodeEq = testScalarLookup<unsigned int>(CNode, YNode, key) || nodeEq;
            // Tests the long long types (they have more complex accessions).
            nodeEq = testScalarLookup<long long>(CNode, YNode, key) || nodeEq;
            nodeEq = testScalarLookup<unsigned long long>(CNode, YNode, key) || nodeEq;
            // Tests floating point types.
            nodeEq = testScalarLookup<double>(CNode, YNode, key) || nodeEq;
            nodeEq = testScalarLookup<float>(CNode, YNode, key) || nodeEq;
            // Tests strings.
            // TODO:: This doesn't compile figure it out later
            // nodeEq = testScalarLookup<const char *>(CNode, YNode, key) || nodeEq;
            nodeEq = testScalarLookup<std::string>(CNode, YNode, key) || nodeEq;
            break;
        // Tests Sequence children.
        case YAML::NodeType::Sequence:
            // Unpacks values for ownership reasons.
            childCNode = CNode.lookup(key);
            childYNode = YNode[key];

            // Passes children to the Sequence tester.
            nodeEq = testSequenceLookup(childCNode, childYNode);
            break;
        case YAML::NodeType::Map:
            // Unpacks values for ownership reasons.
            childCNode = CNode.lookup(key);
            childYNode = YNode[key];

            // Passes it to the Map tester.
            nodeEq = testMapLookup(childCNode, childYNode);
            break;
        // Makes sure the CNode recognizes the value does not exist.
        case YAML::NodeType::Undefined:
            nodeEq = !CNode.exists(key);
            break;
        /* If you got here you shouldn't have because we should always have a
         * valid child type. */
        default:
            std::cout << "!!! UNIT TEST ERROR !!!" << std::endl;
            throw std::invalid_argument("YAML type is invalid: " + std::to_string(TYPE));
            break;
    }
    
    // If there's a failure we print which key failed to construct some error
    // pathway.
    // TODO:: Find a better way to locate where a failure is.
    if (!nodeEq)
    {
        // Prints out the key that failed.
        std::cout << "key: " << key << std::endl;
        // Logs the error in BOOST.
        BOOST_CHECK(nodeEq);
    }

    return nodeEq;
}

// we are only testing things in config
namespace config {
/// @brief Tests the Lookups with files from accelergy-excerises.
BOOST_AUTO_TEST_CASE(testStaticLookups)
{
    // Marker for this test in the printout.
    std::cout << "\n\n\nBeginning Static Lookups Test:\n---" << std::endl;

    // Iterates over all testing dirs.
    for (auto FILEPATH:FILES) 
    {
        // Calculates DIR relative location and extracts file's name.
        std::string DIR = TEST_LOC + FILEPATH.first;
        std::vector<std::string> FILENAMES = FILEPATH.second;

        // Iterates over all filenames.
        for (std::string FILE:FILENAMES)
        {
            // Constructs the full filepath.
            std::string FILEPATH = DIR + FILE;

            // Progress printout regarding which file is currently being tested.
            if (progressReports) std::cout << "Now testing: " + FILEPATH << std::endl;

            // reads the YAML file into CompoundConfig and gets root CCN.
            CompoundConfig cConfig = CompoundConfig({FILEPATH});
            CompoundConfigNode root = cConfig.getRoot();

            // reads in the YAML file independently of CompoundConfig as truth.
            YAML::Node ref = YAML::LoadFile(FILEPATH);

            /* Tests the CompoundConfigNode loading versus the truth, along with
             * the associated lookup functions. */
            BOOST_CHECK(testMapLookup(root, ref));
        }
    }

    std::cout << "Done!" << std::endl;
}

/// @brief Fuzz tests the Setters.
BOOST_AUTO_TEST_CASE(testSettersFuzz)
{
    // Marker for test.
    std::cout << "\n\n\nBeginning Setters Fuzz Test:\n---" << std::endl;
    
    // Seeds the entropy source.
    srand(SEED);
 
    // creates test-bench upon which to test the setter for cConfig
    CompoundConfig cConfig = CompoundConfig("", "yaml"); 
    CompoundConfigNode CNode = cConfig.getRoot();
    
    // creates the reference upon which to compare the test-bench
    YAML::Node YNode = YAML::Node();
 
    for (int test = 0; test < TESTS; test++)
    {
        /* Prints out completion progress of the fuzz test based on # of tests
         * executed. */
        if (progressReports)
        {
            if (test % (TESTS / 20) == 0)
            {
                std::cout << "Progress: " << (float)test / TESTS * 100 << "\% done" << std::endl;
            }
        }

        // Generates a random YAML::NodeType.
        int TYPE = rand() % 5;
        // Generates a random key.
        std::string key = std::to_string(rand() % (TESTS / 1000));
        // Declares the namespace for any value to insert.
        int val;
        int val2;

        // Reports the current values of certain variables at runtime.
        if (runtimeStates)
        {
            std::cout << "---" << std::endl;
            std::cout << "Type: " << TYPE << std::endl;
            std::cout << "Key: " << key << std::endl;
            std::cout << "-\nCNode: " << CNode.getYNode() << std::endl;
            std::cout << "-\nYNode: " << YNode << std::endl;
        }

        switch (TYPE)
        {
            case YAML::NodeType::Null:
                // Instantiates the key.
                CNode.instantiateKey(key);
                // Sets value to Null
                CNode.lookup(key).setScalar(YAML::Null);

                // Writes Scalar to Map.
                YNode[key] = YAML::Node();

                break;
            // Tests Scalars.
            case YAML::NodeType::Scalar:
                // Initializes the Scalar.
                val = rand();

                // Initializes the key.
                CNode.instantiateKey(key);

                // Writes Scalar to Map.
                CNode.lookup(key).setScalar(val);
                YNode[key] = val;

                break;
            case YAML::NodeType::Sequence:
                // Initializes the value to be pushed back.
                val = rand();

                // Instantiates key if it doesn't exist already.
                CNode.instantiateKey(key);

                // Attempts to append value to Sequence.
                CNode.lookup(key).push_back(val);

                /* Attempts to append value to Sequence. Relies on error throws
                 * to determine if it's possible for regular YAML::Node. */
                try {
                    YNode[key].push_back(val);
                } catch (const YAML::BadPushback& e) {}

                break;
            case YAML::NodeType::Map:
                // Generates the map subkey.
                val = rand() % (TESTS / 100);
                // Generates the value to map to.
                val2 = rand();

                // Instantiates the key.
                CNode.instantiateKey(key);

                // Attempts to write to Map only if the key is Null or a Map.
                if (YNode[key].IsMap() || YNode[key].IsNull())
                {
                    // Attempts to write to Map.
                    CNode.lookup(key).instantiateKey(std::to_string(val));
                    CNode.lookup(key).lookup(std::to_string(val)).setScalar(val2);
                    YNode[key][std::to_string(val)] = val2;
                }

                break;
            /* We don't expect undefined values in our code so we will not be
             * implementing this. However, it's left here for completion in case
             * it becomes relevant later on. */
            case YAML::NodeType::Undefined:
                break;
        }
    }

    // Reports the final state of the reference and actual values.
    if (endStateReport)
    {
        std::cout << "Final CNode:" << std::endl;
        std::cout << CNode.getYNode() << std::endl;
        std::cout << "#########################" << std::endl;
        std:: cout << "Final YNode:" << std::endl;
        std::cout << YNode << std::endl;
    }

    // Creates constant YNode clone for testing security.
    const YAML::Node YClone = YAML::Clone(YNode);

    // Creates dummy CompoundConfig for YNode_CNode to reference.
    config::CompoundConfig dummy = config::CompoundConfig("", "yaml");
    config::CompoundConfigNode YNode_CNode = config::CompoundConfigNode(
                                             nullptr, YAML::Clone(YNode), 
                                             &dummy);

    /* Tests YNode against itself, to test symmetry of reads and writes. */
    BOOST_CHECK(testMapLookup(YNode_CNode, YClone));
    /* Tests CNode against reference to ensure the writes were the same. */
    BOOST_CHECK(testMapLookup(CNode, YClone));
    /* Tests CNode accesses with itself as truth to ensure writes can access
     * all the items lookup can. */
    const YAML::Node CNode_YNode = CNode.getYNode();
    BOOST_CHECK(testMapLookup(CNode, CNode_YNode));
    /* Tests YNode_CNode when using CNode_YNode as truth to ensure reads can
     * access all the items writes can. */
    BOOST_CHECK(testMapLookup(YNode_CNode, CNode_YNode));

    std::cout << "Done!" << std::endl;
}

void replicateNode( const config::CompoundConfigNode source,
                    config::CompoundConfigNode sink)
{
    // Replication instructions if the source is a Map.
    if (source.isMap())
    {
        // Fetches Map keys from source.
        std::vector<std::string> keys;
        source.getMapKeys(keys);

        // Iterates through all keys.
        for (std::string key: keys)
        {
            // Declares the key in sink.
            sink.instantiateKey(key);
            // Recurses replication for all key-value pairs.
            replicateNode(source.lookup(key), sink.lookup(key));
        }

        return;
    // Replication instructions if the source is an Array.
    } else if (source.isArray())
    {
        // Fetches the array values.
        std::vector<std::string> arr;
        source.getArrayValue(arr);

        // Writes array values to sink.
        for (std::string elem: arr)
        {
            sink.push_back(elem);
        }

        return;
    // Replication instructions if the source is a List.
    } else if (source.isList())
    {
        // Goes through all elements of the list.
        for (int i = 0; i < source.getLength(); i++)
        {
            // Initializes new value at index.
            sink.push_back(YAML::Node());
            // Recurses replication.
            replicateNode(source[i], sink[i]);
        }
    // Replication instructions if the source is some Scalar/Null
    } else
    {
        // Sets sink value to source value.
        sink.setScalar(source.resolve());
    }
}

/// @brief Tests the ability to replicate a CNode with only CNode methods. 
BOOST_AUTO_TEST_CASE(testReplication)
{
    // Marker for test.
    std::cout << "\n\n\nBeginning Replication Test:\n---" << std::endl;
    // Iterates over all testing dirs.
    for (auto FILEPATH:FILES)
    {
        // Calculates DIR relative location and extracts file's name.
        std::string DIR = TEST_LOC + FILEPATH.first;
        std::vector<std::string> FILENAMES = FILEPATH.second;

        // Iterates over all filenames.
        for (std::string FILE:FILENAMES)
        {
            // Constructs the full filepath.
            std::string FILEPATH = DIR + FILE;

            // Progress printout regarding which file is currently being tested.
            if (progressReports) std::cout << "Now testing: " + FILEPATH << std::endl;

            // Creates the reference.
            config::CompoundConfig ref = config::CompoundConfig({FILEPATH});
            // Creates the replicant.
            config::CompoundConfig rep = config::CompoundConfig("", "yaml");

            // extracts their root CompoundConfigNode
            config::CompoundConfigNode refNode = ref.getRoot();
            config::CompoundConfigNode repNode = rep.getRoot();

            // Creates a deep copy of the node.
            replicateNode(refNode, repNode);

            // Creates the truth source.
            YAML::Node truth = YAML::LoadFile(FILEPATH);

            // Checks reference against truth.
            BOOST_CHECK(testMapLookup(refNode, truth));
            // Checks copy against truth.
            BOOST_CHECK(testMapLookup(repNode, truth));
        }
    }

    std::cout << "Done!" << std::endl;
}
} // namespace config