#define BOOST_TEST_MODULE TestCompoundConfig

#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <compound-config/compound-config.hpp>

// Number of testing cycles to run.
int TESTS = 2000;
// The seed for the entropy source.
uint SEED = 42;
// Changes the max random value to the U_LONG32 random value.
#undef RAND_MAX
#define RAND_MAX = ULONG_LONG_MAX

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

/*!
 * testScalarLookup makes sure for a Map CompoundConfigNode that contains a
 * Scalar value mapped to a key, the contained Scalar value agrees with a 
 * corresponding reference YAML::Node. 
 * 
 * The return value defaults to false if there is a conversion error and does
 * not raise any errors in BOOST or the runtime environment. 
 * 
 * A BOOST error is raised if there is an equality error.
 * 
 * @param CNode The CompoundConfigNode (CCN) thats value is being tested for. It
 *              is expected that a Map CCN is passed in, as CCN's lookupValue
 *              function, which is how it accesses Scalars, only works for
 *              key-value pairs.
 * @param YNode The YAML::Node that serves as a source of truth/reference value
 *              for CNode. It is expected that a YAML::Map Node is passed in to
 *              parallel the CNode structure.
 * @param key   The key corresponding to the Scalar value we want to compare.
 * @return      Returns whether a value at a key is Scalar AND Equal.
 */ 
template <typename T>
bool testScalarLookup(  const config::CompoundConfigNode& CNode, 
                        const YAML::Node& YNode, const std::string& key)
{
    /**
     * First attempt to run the test as normal, keeping an eye out for a YAML
     * conversion error from YAML types to C++ types.
     */
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

    /**
     * if there is a conversion error for the type we're trying to access,
     * return false.
     */
    } catch(const YAML::TypedBadConversion<T>& e) {
        // defaults to false on bad conversion
        return false;
    }
}

// Forward declaration.
bool testMapLookup(config::CompoundConfigNode& CNode, YAML::Node&YNode);
/*!
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
 * @return      Returns whether all the elements in CNode and YNode agree.
 *              Returns true until proven otherwise.
 */
bool testSequenceLookup(config::CompoundConfigNode& CNode, YAML::Node& YNode)
{
    /**
     * Return value namespace + initialization. It defaults to true until an
     * inequality is found.
     */
    bool equal = true;

    // Checks that the children are YAML Sequences or equivalent.
    BOOST_CHECK(CNode.isList() || CNode.isArray());
    BOOST_CHECK(YNode.Type() == YAML::NodeType::Sequence);

    /**
     * If the YNode children are Scalar (if ANY of them are Scalar), it's
     * the equivalent of an Array in CNode. Therefore we should run the Array
     * test execution pathway which triggers the CCN testArrayLookup function.
     */ 
    if (YNode[0].Type() == YAML::NodeType::Scalar)
    {
        // Confirms the CNode is an Array. Raises a BOOST error if not.
        BOOST_CHECK(CNode.isArray());

        // The heap space allocated for the CCN to dump its Array values into.
        std::vector<std::string> actual;

        /**
         * Fetches Array, should always work if our previous code works as 
         * intended (i.e. the assumption, which should be the standard, that if 
         * ANY child of the YNode is a Scalar, the CNode is an Array). If not,
         * we raise a BOOST error.
         */
        BOOST_CHECK(CNode.getArrayValue(actual));

        /**
         * Iterates over the YNode elements and compares them to the 
         * corresponding fetched element. Update return value to false if an
         * inequality is found.
         */
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
                BOOST_CHECK(childYNode.Type() == YAML::NodeType::Map);

                /**
                 * Checks equality in the child nodes via testMapLookup. This
                 * method only works because values are always labeled.
                 */ 
                equal = equal && testMapLookup(childCNode, childYNode);
            }
        }
    }

    return equal;
}

// Forward declaration.
bool mapNodeEq( config::CompoundConfigNode CNode, YAML::Node YNode, 
                const std::string& key, YAML::NodeType::value TYPE);
/*!
 * testMapLookup makes sure for a Map CompoundConfigNode, the key-value pairs in 
 * the Map agree with a reference YAML::Node.
 * 
 * The return value defaults to true until an element inequality is found.
 * 
 * @param CNode The CompoundConfigNode whose key-value pairs are being equality
 *              tested. A Map CCN is expected to be passed in.
 * @param YNode The Map YAML::Node which serves as a source of truth/reference 
 *              the CCN key-value pairs are being compared against.
 * @return      Returns whether the key-value pairs in CNode and YNode agree.
 *              returns true until proven otherwise.
 */
bool testMapLookup(config::CompoundConfigNode& CNode, YAML::Node&YNode)
{
    // Defines the return value namespace and instantiates it as true.
    bool equal = true;

    // Iterates over all key-value pairs in YNode and compares to CNode.
    for (auto nodeMapPair: YNode)
    {
        // Extracts the key from YNode.
        const std::string key = nodeMapPair.first.as<std::string>();
        /**
         * Checks the value at key are equal. Note that we did not unpack the
         * value at the key for either the CNode or the YNode. This is because
         * if the value at the key is a Scalar, we need to preserve the packing
         * given CompoundConfigNode can only access packed Scalars. However,
         * since a Map value can be anything, we need to abstract the logic
         * of testing for equality here to another function as the CCN expected
         * behavior is different depending on the value type.
         */
        equal = equal && mapNodeEq(CNode, YNode, key, nodeMapPair.second.Type());
    }

    return equal;
}

/*! 
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
 * @return      Returns whether the CNode is equal to the reference. Defaults to
 *              false.
 */
bool mapNodeEq( config::CompoundConfigNode CNode, YAML::Node YNode, 
                const std::string& key, YAML::NodeType::value TYPE)
{
    // Declares the namespace of the return value. Instantiates to false.
    bool nodeEq = false;

    /**
     * Declares the namespace of the child nodes we'd unpack if we're accessing
     * a Sequence or Map for ownership reasons. Switch statements don't like
     * declaring variables depending on which execution pathway is taken.
     */
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
        /**
         * If you got here you shouldn't have because we should always have a
         * valid child type.
         */
        default:
            std::cout << "!!! UNIT TEST ERROR !!!" << std::endl;
            throw std::invalid_argument("YAML type is invalid: " + TYPE);
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
// tests the lookup functions when reading in from file
BOOST_AUTO_TEST_CASE(testStaticLookups)
{
    // marker for test
    std::cout << "\n\n\nBeginning Static Lookups Test:\n---" << std::endl;
    // goes through all testing dirs
    for (auto FILEPATH:FILES) 
    {
        // calculates DIR relative location and extracts file's name
        std::string DIR = TEST_LOC + FILEPATH.first;
        std::vector<std::string> FILENAMES = FILEPATH.second;

        for (std::string FILE:FILENAMES)
        {
            // calculates filepath
            std::string FILEPATH = DIR + FILE;
            // debug printing info
            std::cout << "Now testing: " + FILEPATH << std::endl;
            // reads the YAML file into CompoundConfig and gets root
            CompoundConfig cConfig = CompoundConfig({FILEPATH});
            CompoundConfigNode root = cConfig.getRoot();
            // reads in the YAML file independently of CompoundConfig to serve as test reference
            YAML::Node ref = YAML::LoadFile(FILEPATH);

            // tests the entire file
            BOOST_CHECK(testMapLookup(root, ref));
        }
    }

    std::cout << "Done!" << std::endl;
}

// tests the ability to set correctly
BOOST_AUTO_TEST_CASE(testSetters)
{
    // marker for test
    std::cout << "\n\n\nBeginning Static Lookups Test:\n---" << std::endl;
 
    // creates test-bench upon which to test the setter for cConfig
    CompoundConfig cConfig = CompoundConfig("", "yaml"); 
    CompoundConfigNode CNode = cConfig.getRoot();
    
    // creates the reference upon which to compare the test-bench
    YAML::Node YNode = YAML::Node();
 
    for (int test = 0; test < TESTS; test++)
    {
        if (test % (TESTS / 20) == 0)
        {
            std::cout << "Progress: " << (float)test / TESTS * 100 << "% done" << std::endl;
        }

        // Generates a random YAML::NodeType.
        int TYPE = rand() % 5;
        // Generates a random key.
        std::string key = std::to_string(rand());
        // Declares the namespace for any value to insert.
        int val;

        switch (TYPE)
        {
            case YAML::NodeType::Null:
                break;
            // Tests Scalars.
            case YAML::NodeType::Scalar:
                // Initializes the Scalar.
                val = rand();

                // Writes Scalar to Map.
                CNode.setValue(key, val);
                YNode[key] = val;

                break;
            case YAML::NodeType::Sequence:
                break;
            case YAML::NodeType::Map:
                break;
            case YAML::NodeType::Undefined:
                break;
        }
    }

    BOOST_CHECK(testMapLookup(CNode, YNode));
    std::cout << "Done!" << std::endl;
}

// tests the ability to read out correctly from sets
BOOST_AUTO_TEST_CASE(testDynamicLookups)
{
    std:: cout << "not yet implemented" << std::endl;
}
} // namespace config